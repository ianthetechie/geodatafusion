use std::any::Any;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, ListArray, UInt32Array, new_empty_array};
use arrow_buffer::{NullBuffer, NullBufferBuilder, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::arrow::compute::{concat, interleave, take};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion::logical_expr::{
    Accumulator, AggregateUDFImpl, Documentation, EmitTo, GroupsAccumulator, Signature,
};
use datafusion::scalar::ScalarValue;
use geoarrow_array::GeoArrowArray;
use geoarrow_array::array::from_arrow_array;
use geoarrow_array::builder::WkbBuilder;
use geoarrow_array::cast::{from_wkb, to_wkb};
use geoarrow_schema::{CoordType, GeoArrowType, GeometryType, Metadata, WkbType};

use crate::data_types::any_single_geometry_type_input;
use crate::error::GeoDataFusionResult;

// TODO: Do we really need this sort of hand-rolled WKB? If so, can we put it in its own module?
// TODO: Does this / should this work with native arrow geometries?

/// WKB byte-order marker for little-endian (NDR) encoding — the encoding geoarrow's `to_wkb` emits.
const WKB_NDR: u8 = 1;

// ISO-WKB type codes are `base + dimension_offset`. The dimension offset is 0 (XY), 1000 (XYZ),
// 2000 (XYM), or 3000 (XYZM); the base identifies the kind: Point=1, LineString=2, Polygon=3,
// MultiPoint=4, MultiLineString=5, MultiPolygon=6, GeometryCollection=7. The MULTI* of an atomic
// base is therefore `base + ATOMIC_TO_MULTI`.
const DIM_MODULUS: u32 = 1000;
const MAX_ATOMIC_BASE: u32 = 3;
const ATOMIC_TO_MULTI: u32 = 3;
const GEOMETRY_COLLECTION_BASE: u32 = 7;

/// Read a WKB geometry's ISO type code (`base + dimension_offset`).
///
/// `wkb` is always a valid ISO-WKB blob (≥5 bytes) produced by geoarrow's `to_wkb`, so the slicing
/// below cannot panic in practice.
fn member_type_code(wkb: &[u8]) -> u32 {
    let code: [u8; 4] = wkb[1..5]
        .try_into()
        .expect("WKB header has a 4-byte type code");
    if wkb[0] == WKB_NDR {
        u32::from_le_bytes(code)
    } else {
        u32::from_be_bytes(code)
    }
}

/// Assemble a single MULTI*/GEOMETRYCOLLECTION as ISO-WKB from member WKBs.
///
/// Per PostGIS `ST_Collect`: homogeneous atomic members (all Point, all LineString, or all Polygon)
/// yield the matching `MULTI*`; mixed base types, or members that are themselves multis/collections,
/// yield a `GEOMETRYCOLLECTION`. Members are embedded verbatim (no dedup or dissolve), so Z/M are
/// preserved.
///
/// All members must share one coordinate dimension. geoarrow's `Geometry` type stores each
/// collection element under a single dimension, so mixing dimensions (e.g. XY with XYZ) is
/// unrepresentable and returns an error rather than panicking in the downstream array builder.
///
/// `members` must be non-empty.
fn build_container_wkb(members: &[&[u8]]) -> GeoDataFusionResult<Vec<u8>> {
    let code0 = member_type_code(members[0]);
    let (base0, dim0) = (code0 % DIM_MODULUS, code0 - code0 % DIM_MODULUS);

    if members.iter().any(|m| {
        let code = member_type_code(m);
        code - code % DIM_MODULUS != dim0
    }) {
        return Err(DataFusionError::NotImplemented(
            "ST_Collect cannot combine geometries of differing coordinate dimensions (e.g. XY and \
             XYZ)"
                .to_string(),
        )
        .into());
    }

    // Dimensions are now known uniform, so homogeneity reduces to a shared atomic base type.
    let homogeneous_atomic = (1..=MAX_ATOMIC_BASE).contains(&base0)
        && members
            .iter()
            .all(|m| member_type_code(m) % DIM_MODULUS == base0);

    let container_base = if homogeneous_atomic {
        base0 + ATOMIC_TO_MULTI
    } else {
        GEOMETRY_COLLECTION_BASE
    };
    let container_code = container_base + dim0;

    let total: usize = members.iter().map(|m| m.len()).sum();
    // Header is the endianness byte + u32 type code + u32 member count.
    let mut buf = Vec::with_capacity(9 + total);
    buf.push(WKB_NDR);
    buf.extend_from_slice(&container_code.to_le_bytes());
    buf.extend_from_slice(&(members.len() as u32).to_le_bytes());
    for m in members {
        buf.extend_from_slice(m);
    }
    Ok(buf)
}

/// Encode an input geometry array to its WKB bytes as a contiguous Arrow `BinaryArray`.
///
/// This is the one allocation per input batch: geoarrow packs every geometry's WKB into a single
/// Arrow buffer with offsets, so members can later be referenced as zero-copy `&[u8]` slices (via
/// `BinaryArray::value`) rather than copied out one `Vec` at a time.
fn geom_to_wkb_binary(values: &ArrayRef, input_field: &FieldRef) -> GeoDataFusionResult<ArrayRef> {
    let geo = from_arrow_array(values, input_field)?;
    let wkb = to_wkb::<i32>(geo.as_ref())?;
    // Cheap: `GenericBinaryArray` clone shares the underlying (Arc-backed) Arrow buffers.
    Ok(Arc::new(wkb.inner().clone()))
}

/// Assemble the mixed-`Geometry` output array from one container WKB per output row.
///
/// `containers` yields one entry per row in group order — `Some(wkb)` for a non-empty group, `None`
/// for an empty/all-NULL group (rendered as a NULL geometry). Every container goes into a single
/// `WkbArray`, so the (comparatively expensive) WKB→`Geometry` parse runs exactly once for the whole
/// output instead of once per group.
fn build_geometry_array(
    containers: impl IntoIterator<Item = Option<Vec<u8>>>,
    metadata: Arc<Metadata>,
    coord_type: CoordType,
) -> GeoDataFusionResult<ArrayRef> {
    let geom_type = GeometryType::new(metadata.clone()).with_coord_type(coord_type);
    let mut builder = WkbBuilder::<i32>::new(WkbType::new(metadata));
    for container in containers {
        builder.push_wkb(container.as_deref())?;
    }
    let wkb = builder.finish();
    let out = from_wkb(&wkb, GeoArrowType::Geometry(geom_type))?;
    Ok(out.to_array_ref())
}

/// Convert a selection filter into a `NullBuffer` marking filtered-out rows (value `false`, or a NULL
/// filter slot) as null — used to drop those rows from `convert_to_state` output.
fn filter_to_null_buffer(filter: &BooleanArray) -> Option<NullBuffer> {
    let (keep, filter_nulls) = filter.clone().into_parts();
    NullBuffer::union(Some(&NullBuffer::new(keep)), filter_nulls.as_ref())
}

/// `ST_Collect` aggregate: collects a set of geometries into one MULTI*/GEOMETRYCOLLECTION.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct CollectAggregate {
    coord_type: CoordType,
}

impl CollectAggregate {
    pub fn new(coord_type: CoordType) -> Self {
        Self { coord_type }
    }
}

impl Default for CollectAggregate {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

impl AggregateUDFImpl for CollectAggregate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    // TODO: ST_CollectAgg?
    fn name(&self) -> &str {
        "st_collect"
    }

    fn signature(&self) -> &Signature {
        any_single_geometry_type_input()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(DataFusionError::Internal("return_type".to_string()))
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let metadata = Arc::new(Metadata::try_from(arg_fields[0].as_ref()).unwrap_or_default());
        Ok(Arc::new(
            GeometryType::new(metadata)
                .with_coord_type(self.coord_type)
                .to_field("", true),
        ))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        // The partial state is the accumulated member geometries, serialized as a list of WKB blobs.
        Ok(vec![Arc::new(Field::new(
            format!("{}[wkbs]", args.name),
            DataType::List(Arc::new(Field::new_list_field(DataType::Binary, true))),
            true,
        ))])
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let input_field = acc_args.exprs[0].return_field(acc_args.schema)?;
        let metadata = Arc::new(Metadata::try_from(input_field.as_ref()).unwrap_or_default());
        Ok(Box::new(CollectAccumulator {
            batches: Vec::new(),
            input_field,
            metadata,
            coord_type: self.coord_type,
        }))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        // Use the vectorized `GroupsAccumulator` for plain `GROUP BY`. DISTINCT and ORDER BY have no
        // defined meaning for `ST_Collect` and fall back to the simple `Accumulator`, preserving today's
        // (non-deduping, order-insensitive) behavior.
        !args.is_distinct && args.order_bys.is_empty()
    }

    fn create_groups_accumulator(
        &self,
        acc_args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let input_field = acc_args.exprs[0].return_field(acc_args.schema)?;
        let metadata = Arc::new(Metadata::try_from(input_field.as_ref()).unwrap_or_default());
        Ok(Box::new(CollectGroupsAccumulator::new(
            input_field,
            metadata,
            self.coord_type,
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Aggregate that turns a set of geometries (often involving a GROUP BY clause) into a single collection without any changes. \
                 Unlike ST_Union, the included geometries are not processed in any way (e.g. no merging of overlapping geometries, nor inserting nodes at LineString intersections). \
                 Returns a MULTI* if all inputs share a single atomic type (Point/LineString/Polygon), otherwise a GEOMETRYCOLLECTION. \
                 NULL inputs are skipped; an all-NULL or empty group yields NULL.
                 Z and M are preserved, but all inputs must be of the same coordinate dimension
                 (mixing e.g. XY and XYZ is an error). \
                 This is the inverse of ST_Dump.",
                "ST_Collect(geom)",
            )
            .with_argument("geom", "geometry")
            .build()
        }))
    }
}

#[derive(Debug)]
struct CollectAccumulator {
    /// Accumulated WKB members held as contiguous Arrow `BinaryArray` buffers — one per `update`/
    /// `merge` batch — and referenced by slice at evaluate time, so no individual geometry is copied.
    batches: Vec<ArrayRef>,
    input_field: FieldRef,
    metadata: Arc<Metadata>,
    coord_type: CoordType,
}

impl CollectAccumulator {
    fn update_inner(&mut self, values: &[ArrayRef]) -> GeoDataFusionResult<()> {
        self.batches
            .push(geom_to_wkb_binary(&values[0], &self.input_field)?);
        Ok(())
    }

    fn evaluate_inner(&self) -> GeoDataFusionResult<ScalarValue> {
        let mut members: Vec<&[u8]> = Vec::new();
        for batch in &self.batches {
            members.extend(batch.as_binary::<i32>().iter().flatten());
        }
        let container = if members.is_empty() {
            None
        } else {
            Some(build_container_wkb(&members)?)
        };
        let arr = build_geometry_array(
            std::iter::once(container),
            self.metadata.clone(),
            self.coord_type,
        )?;
        Ok(ScalarValue::try_from_array(arr.as_ref(), 0)?)
    }
}

impl Accumulator for CollectAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Ok(self.update_inner(values)?)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.evaluate_inner()?)
    }

    /// Partial state is the members as a single-row `List<Binary>`. The member buffers are
    /// concatenated once (sharing the input data) rather than re-cloned blob-by-blob.
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let values: ArrayRef = if self.batches.is_empty() {
            new_empty_array(&DataType::Binary)
        } else {
            let refs: Vec<&dyn Array> = self.batches.iter().map(|b| b.as_ref()).collect();
            concat(&refs)?
        };
        let offsets = OffsetBuffer::<i32>::from_lengths([values.len()]);
        let field = Arc::new(Field::new_list_field(DataType::Binary, true));
        let list = ListArray::new(field, offsets, values, None);
        Ok(vec![ScalarValue::List(Arc::new(list))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // Each non-NULL list row contributes its backing `BinaryArray` slice as a batch; NULL members
        // within are skipped when slices are gathered at evaluate time.
        for inner in states[0].as_list::<i32>().iter().flatten() {
            self.batches.push(inner);
        }
        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.batches.capacity() * std::mem::size_of::<ArrayRef>()
            + self
                .batches
                .iter()
                .map(|b| b.get_array_memory_size())
                .sum::<usize>()
    }
}

/// Vectorized `GroupsAccumulator` for more efficient high-cardinality grouped aggregates.
#[derive(Debug)]
struct CollectGroupsAccumulator {
    input_field: FieldRef,
    metadata: Arc<Metadata>,
    coord_type: CoordType,
    /// WKB source arrays — input geometries (from `update_batch`) or list-backing values (from
    /// `merge_batch`) — all `BinaryArray`s referenced by `batch_entries`.
    batches: Vec<ArrayRef>,
    /// Per-batch `(group_idx, row_idx)` pairs for rows that survived filtering and were non-NULL.
    batch_entries: Vec<Vec<(u32, u32)>>,
    num_groups: usize,
}

impl CollectGroupsAccumulator {
    fn new(input_field: FieldRef, metadata: Arc<Metadata>, coord_type: CoordType) -> Self {
        Self {
            input_field,
            metadata,
            coord_type,
            batches: Vec::new(),
            batch_entries: Vec::new(),
            num_groups: 0,
        }
    }

    /// Counting-sort the retained entries for groups `[0, emit_groups)` into group order.
    ///
    /// Returns `(offsets, order)`: `offsets` has `emit_groups + 1` entries delimiting each group's
    /// run, and `order[k] = (batch_idx, row_idx)` is the source of the k-th member in group order. A
    /// group with no members has an empty run (`offsets[g] == offsets[g + 1]`).
    fn group_order(&self, emit_groups: usize) -> (Vec<i32>, Vec<(usize, usize)>) {
        let mut offsets = vec![0i32; emit_groups + 1];
        for entries in &self.batch_entries {
            for &(g, _) in entries {
                let g = g as usize;
                if g < emit_groups {
                    offsets[g + 1] += 1;
                }
            }
        }
        for i in 0..emit_groups {
            offsets[i + 1] += offsets[i];
        }
        let total = offsets[emit_groups] as usize;
        let mut write_pos: Vec<i32> = offsets[..emit_groups].to_vec();
        let mut order = vec![(0usize, 0usize); total];
        for (bi, entries) in self.batch_entries.iter().enumerate() {
            for &(g, r) in entries {
                let g = g as usize;
                if g < emit_groups {
                    let wp = write_pos[g] as usize;
                    order[wp] = (bi, r as usize);
                    write_pos[g] += 1;
                }
            }
        }
        (offsets, order)
    }

    /// Release state for the groups just emitted per `emit_to`.
    fn reset_after_emit(&mut self, emit_to: EmitTo) -> Result<()> {
        match emit_to {
            EmitTo::All => {
                self.batches.clear();
                self.batch_entries.clear();
                self.num_groups = 0;
            }
            EmitTo::First(n) => self.compact_retained_state(n)?,
        }
        Ok(())
    }

    /// Rebuild state retaining only groups `>= emit_groups` (renumbered to start at 0), used by
    /// `EmitTo::First` under memory pressure. Fully-emitted batches are dropped and mixed batches are
    /// compacted via `take` so retained rows no longer pin whole input arrays. Ported from `array_agg`.
    fn compact_retained_state(&mut self, emit_groups: usize) -> Result<()> {
        let emit_groups = emit_groups as u32;
        let old_batches = std::mem::take(&mut self.batches);
        let old_batch_entries = std::mem::take(&mut self.batch_entries);
        for (batch, entries) in old_batches.into_iter().zip(old_batch_entries) {
            let retained_len = entries.iter().filter(|(g, _)| *g >= emit_groups).count();
            if retained_len == 0 {
                continue;
            }
            if retained_len == entries.len() {
                let mut retained_entries = entries;
                for (g, _) in &mut retained_entries {
                    *g -= emit_groups;
                }
                self.batches.push(batch);
                self.batch_entries.push(retained_entries);
                continue;
            }
            let mut retained_entries = Vec::with_capacity(retained_len);
            let mut retained_rows = Vec::with_capacity(retained_len);
            for (g, r) in entries {
                if g >= emit_groups {
                    retained_entries.push((g - emit_groups, retained_rows.len() as u32));
                    retained_rows.push(r);
                }
            }
            let batch = if retained_len == batch.len() {
                batch
            } else {
                take(batch.as_ref(), &UInt32Array::from(retained_rows), None)?
            };
            self.batches.push(batch);
            self.batch_entries.push(retained_entries);
        }
        self.num_groups -= emit_groups as usize;
        Ok(())
    }
}

impl GroupsAccumulator for CollectGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let wkb = geom_to_wkb_binary(&values[0], &self.input_field)?;
        self.num_groups = self.num_groups.max(total_num_groups);
        let mut entries = Vec::new();
        for (row, &group) in group_indices.iter().enumerate() {
            if let Some(filter) = opt_filter
                && (filter.is_null(row) || !filter.value(row))
            {
                continue;
            }
            // NULL geometries contribute nothing (matches ST_Collect semantics).
            if wkb.is_null(row) {
                continue;
            }
            entries.push((group as u32, row as u32));
        }
        if !entries.is_empty() {
            self.batches.push(wkb);
            self.batch_entries.push(entries);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let emit_groups = match emit_to {
            EmitTo::All => self.num_groups,
            EmitTo::First(n) => n,
        };
        let (offsets, order) = self.group_order(emit_groups);
        let mut members: Vec<&[u8]> = Vec::new();
        let mut containers: Vec<Option<Vec<u8>>> = Vec::with_capacity(emit_groups);
        for g in 0..emit_groups {
            let (start, end) = (offsets[g] as usize, offsets[g + 1] as usize);
            if start == end {
                containers.push(None);
                continue;
            }
            members.clear();
            for &(bi, ri) in &order[start..end] {
                members.push(self.batches[bi].as_binary::<i32>().value(ri));
            }
            containers.push(Some(build_container_wkb(&members)?));
        }
        let array = build_geometry_array(containers, self.metadata.clone(), self.coord_type)?;
        self.reset_after_emit(emit_to)?;
        Ok(array)
    }

    /// Partial state: a `List<Binary>` with one list of WKB members per group, in group order.
    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let emit_groups = match emit_to {
            EmitTo::All => self.num_groups,
            EmitTo::First(n) => n,
        };
        let (offsets, order) = self.group_order(emit_groups);
        let values: ArrayRef = if order.is_empty() {
            new_empty_array(&DataType::Binary)
        } else {
            let sources: Vec<&dyn Array> = self.batches.iter().map(|b| b.as_ref()).collect();
            interleave(&sources, &order)?
        };
        let mut nulls = NullBufferBuilder::new(emit_groups);
        for g in 0..emit_groups {
            if offsets[g] == offsets[g + 1] {
                nulls.append_null();
            } else {
                nulls.append_non_null();
            }
        }
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
        let field = Arc::new(Field::new_list_field(DataType::Binary, true));
        let list = ListArray::new(field, offsets, values, nulls.finish());
        self.reset_after_emit(emit_to)?;
        Ok(vec![Arc::new(list)])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let list = values[0].as_list::<i32>();
        self.num_groups = self.num_groups.max(total_num_groups);
        let list_offsets = list.offsets();
        let mut entries = Vec::new();
        for (row, &group) in group_indices.iter().enumerate() {
            if list.is_null(row) {
                continue;
            }
            let start = list_offsets[row] as u32;
            let end = list_offsets[row + 1] as u32;
            for pos in start..end {
                entries.push((group as u32, pos));
            }
        }
        if !entries.is_empty() {
            self.batches.push(Arc::clone(list.values()));
            self.batch_entries.push(entries);
        }
        Ok(())
    }

    /// Treat each input row as its own group: emit a `List<Binary>` whose i-th list holds just row i's
    /// WKB (or NULL when filtered out or NULL). DataFusion uses this to bypass the group hash table
    /// when cardinality approaches the row count — the common case here.
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let wkb = geom_to_wkb_binary(&values[0], &self.input_field)?;
        let len = wkb.len();
        let offsets = OffsetBuffer::<i32>::from_lengths(std::iter::repeat_n(1usize, len));
        let filter_nulls = opt_filter.and_then(filter_to_null_buffer);
        let nulls = NullBuffer::union(filter_nulls.as_ref(), wkb.nulls());
        let field = Arc::new(Field::new_list_field(DataType::Binary, true));
        let list = ListArray::new(field, offsets, wkb, nulls);
        Ok(vec![Arc::new(list)])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self
                .batches
                .iter()
                .map(|b| b.get_array_memory_size())
                .sum::<usize>()
            + self
                .batch_entries
                .iter()
                .map(|e| e.capacity() * std::mem::size_of::<(u32, u32)>())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod test {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use datafusion::prelude::{SessionConfig, SessionContext};

    use super::*;
    use crate::udf::native::accessors::Dump;
    use crate::udf::native::io::{AsText, GeomFromText};

    fn register(ctx: &SessionContext) {
        ctx.register_udaf(CollectAggregate::default().into());
        ctx.register_udf(GeomFromText::default().into());
        ctx.register_udf(AsText.into());
        ctx.register_udf(Dump::default().into());
    }

    fn ctx() -> SessionContext {
        let ctx = SessionContext::new();
        register(&ctx);
        ctx
    }

    /// A context whose plans split aggregation across batches/partitions, exercising the
    /// partial→merge→final path (`update_batch`/`state`/`merge_batch`/`evaluate`).
    fn ctx_two_phase() -> SessionContext {
        let config = SessionConfig::new()
            .with_batch_size(1)
            .with_target_partitions(4);
        let ctx = SessionContext::new_with_config(config);
        register(&ctx);
        ctx
    }

    /// Build a geometry array (and its field) from WKT literals via `ST_GeomFromText`, for driving a
    /// `CollectGroupsAccumulator` directly. A `NULL` WKT yields a NULL geometry.
    async fn geom_array(ctx: &SessionContext, wkts: &[Option<&str>]) -> (ArrayRef, FieldRef) {
        let values = wkts
            .iter()
            .map(|w| match w {
                Some(w) => format!("('{w}')"),
                None => "(CAST(NULL AS TEXT))".to_string(),
            })
            .collect::<Vec<_>>()
            .join(",");
        let df = ctx
            .sql(&format!(
                "SELECT ST_GeomFromText(w) AS g FROM (VALUES {values}) AS t(w)"
            ))
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let array = batch.column_by_name("g").unwrap().clone();
        let field = Arc::new(batch.schema().field_with_name("g").unwrap().clone());
        (array, field)
    }

    /// Render a mixed-`Geometry` output array (as produced by `evaluate`) to per-row WKT, mapping NULL
    /// rows to the literal `"NULL"`.
    fn geom_wkts(array: &ArrayRef, metadata: Arc<Metadata>, coord_type: CoordType) -> Vec<String> {
        let geom_type = GeometryType::new(metadata).with_coord_type(coord_type);
        let field = geom_type.to_field("", true);
        let geo = from_arrow_array(array, &field).unwrap();
        let wkt = geoarrow_array::cast::to_wkt::<i32>(geo.as_ref())
            .unwrap()
            .to_array_ref();
        let strings = wkt.as_string::<i32>();
        (0..array.len())
            .map(|i| {
                if strings.is_null(i) {
                    "NULL".to_string()
                } else {
                    strings.value(i).to_string()
                }
            })
            .collect()
    }

    fn collect_meta(field: &FieldRef) -> Arc<Metadata> {
        Arc::new(Metadata::try_from(field.as_ref()).unwrap_or_default())
    }

    /// Run `sql` and return the (nullable) first text cell of the first row.
    async fn text1(ctx: &SessionContext, sql: &str) -> Option<String> {
        let df = ctx.sql(sql).await.unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let col = batch.column(0).as_string::<i32>();
        (!col.is_null(0)).then(|| col.value(0).to_string())
    }

    /// Collect a single VALUES list of WKT literals (single group).
    async fn collect_wkts(ctx: &SessionContext, wkts: &[&str]) -> Option<String> {
        let values = wkts
            .iter()
            .map(|w| format!("('{w}')"))
            .collect::<Vec<_>>()
            .join(",");
        text1(
            ctx,
            &format!(
                "SELECT ST_AsText(ST_Collect(ST_GeomFromText(w))) FROM (VALUES {values}) AS t(w)"
            ),
        )
        .await
    }

    /// Homogeneous atomic inputs collect into the matching MULTI*.
    #[tokio::test]
    async fn test_collect_homogeneous() {
        let ctx = ctx();
        // PostGIS doc example shape: points -> MultiPoint.
        assert_eq!(
            collect_wkts(&ctx, &["POINT(0 0)", "POINT(1 1)", "POINT(2 2)"])
                .await
                .as_deref(),
            Some("MULTIPOINT((0 0),(1 1),(2 2))")
        );
        // PostGIS doc example: lines -> MultiLineString.
        assert_eq!(
            collect_wkts(&ctx, &["LINESTRING(1 2,3 4)", "LINESTRING(3 4,4 5)"])
                .await
                .as_deref(),
            Some("MULTILINESTRING((1 2,3 4),(3 4,4 5))")
        );
        // A single input still becomes a MULTI* of one.
        assert_eq!(
            collect_wkts(&ctx, &["POINT(5 5)"]).await.as_deref(),
            Some("MULTIPOINT((5 5))")
        );
    }

    /// Mixed types (or members that are themselves collections) collect into a GeometryCollection.
    #[tokio::test]
    async fn test_collect_mixed_to_gc() {
        let ctx = ctx();
        assert_eq!(
            collect_wkts(&ctx, &["POINT(0 0)", "LINESTRING(1 1,2 2)"])
                .await
                .as_deref(),
            Some("GEOMETRYCOLLECTION(POINT(0 0),LINESTRING(1 1,2 2))")
        );
    }

    /// Mixing coordinate dimensions in a single collection is unrepresentable in geoarrow's
    /// `Geometry` type, so it surfaces as a clean error rather than panicking the query.
    #[tokio::test]
    async fn test_collect_mixed_dimensions_errors() {
        let ctx = ctx();
        let df = ctx
            .sql(
                "SELECT ST_AsText(ST_Collect(ST_GeomFromText(w))) \
                 FROM (VALUES ('POINT(0 0)'), ('POINT Z(1 1 1)')) AS t(w)",
            )
            .await
            .unwrap();
        let err = df.collect().await.unwrap_err();
        assert!(
            err.to_string().contains("dimension"),
            "expected a dimension error, got: {err}"
        );
    }

    /// An empty member is embedded verbatim (see the MULTIPOINT/MULTIPOLYGON empty-member caveats —
    /// MULTILINESTRING is the safe vehicle for this). geoarrow renders an empty member as `()`.
    #[tokio::test]
    async fn test_collect_empty_member() {
        let ctx = ctx();
        assert_eq!(
            collect_wkts(&ctx, &["LINESTRING(0 0,1 1)", "LINESTRING EMPTY"])
                .await
                .as_deref(),
            Some("MULTILINESTRING((0 0,1 1),())")
        );
    }

    /// Z is preserved (a GEOS round-trip would strip M; this native path keeps both).
    #[tokio::test]
    async fn test_collect_preserves_z() {
        let ctx = ctx();
        assert_eq!(
            collect_wkts(&ctx, &["POINT Z(1 2 3)", "POINT Z(1 2 4)"])
                .await
                .as_deref(),
            Some("MULTIPOINT Z((1 2 3),(1 2 4))")
        );
    }

    /// NULL inputs are skipped; an all-NULL group yields NULL.
    #[tokio::test]
    async fn test_collect_nulls() {
        let ctx = ctx();
        let one_null = text1(
            &ctx,
            "SELECT ST_AsText(ST_Collect(ST_GeomFromText(w))) \
             FROM (VALUES ('POINT(0 0)'), (CAST(NULL AS TEXT)), ('POINT(1 1)')) AS t(w)",
        )
        .await;
        assert_eq!(one_null.as_deref(), Some("MULTIPOINT((0 0),(1 1))"));

        let all_null = text1(
            &ctx,
            "SELECT ST_AsText(ST_Collect(ST_GeomFromText(w))) \
             FROM (VALUES (CAST(NULL AS TEXT))) AS t(w)",
        )
        .await;
        assert_eq!(all_null, None);
    }

    /// Aggregate with GROUP BY — the canonical PostGIS usage shape.
    #[tokio::test]
    async fn test_collect_group_by() {
        let ctx = ctx();
        let df = ctx
            .sql(
                "SELECT k, ST_AsText(ST_Collect(ST_GeomFromText(w))) AS g \
                 FROM (VALUES (1,'POINT(0 0)'),(1,'POINT(1 1)'),(2,'POINT(9 9)')) AS t(k,w) \
                 GROUP BY k ORDER BY k",
            )
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let g = batch.column_by_name("g").unwrap().as_string::<i32>();
        assert_eq!(g.value(0), "MULTIPOINT((0 0),(1 1))");
        assert_eq!(g.value(1), "MULTIPOINT((9 9))");
    }

    /// Round-trip: `ST_Collect` over the components of `ST_Dump` reconstructs flat multis.
    #[tokio::test]
    async fn test_roundtrip_dump_collect() {
        let ctx = ctx();

        // Extract the component WKTs that ST_Dump produces for `wkt`.
        async fn dump_component_wkts(ctx: &SessionContext, wkt: &str) -> Vec<String> {
            let df = ctx
                .sql(&format!("SELECT ST_Dump(ST_GeomFromText('{wkt}'))"))
                .await
                .unwrap();
            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let structs = batch.column(0).as_list::<i32>().value(0);
            let structs = structs.as_struct();
            let DataType::Struct(fields) = structs.data_type() else {
                panic!("ST_Dump returns a list of structs");
            };
            let geo = from_arrow_array(structs.column(1).as_ref(), fields[1].as_ref()).unwrap();
            let wkt_ref = geoarrow_array::cast::to_wkt::<i32>(geo.as_ref())
                .unwrap()
                .to_array_ref();
            let wkts = wkt_ref.as_string::<i32>();
            (0..structs.len())
                .map(|i| wkts.value(i).to_string())
                .collect()
        }

        for orig in [
            "MULTIPOINT(0 0,1 1,2 2)",
            "MULTILINESTRING((0 0,1 1),(2 2,3 3))",
            "MULTIPOLYGON(((0 0,0 1,1 1,1 0,0 0)),((10 10,10 20,20 20,20 10,10 10)))",
        ] {
            let parts = dump_component_wkts(&ctx, orig).await;
            let reconstructed =
                collect_wkts(&ctx, &parts.iter().map(String::as_str).collect::<Vec<_>>()).await;
            // Compare against the canonical (ST_AsText-normalized) form of the original, since
            // ST_AsText parenthesizes MultiPoint parts.
            let expected = text1(
                &ctx,
                &format!("SELECT ST_AsText(ST_GeomFromText('{orig}'))"),
            )
            .await;
            assert_eq!(reconstructed, expected, "round-trip failed for {orig}");
        }
    }

    /// Collection-member parity with PostGIS 3.6.3 / GEOS 3.13.1: a member that is itself a MULTI*
    /// (or a different base type) makes ST_Collect return a GEOMETRYCOLLECTION with members embedded
    /// verbatim — never flattened — exactly as PostGIS does.
    ///
    /// Verify by pasting into psql ("postgresql://osm:osm@localhost:5432/osm"), e.g.:
    ///   SELECT ST_AsText(ST_Collect(ST_GeomFromText(w)))
    ///   FROM (VALUES ('LINESTRING(0 0,1 1)'),('MULTILINESTRING((2 2,3 3),(4 4,5 5))')) AS t(w);
    #[tokio::test]
    async fn test_collect_parity_collection_members() {
        let ctx = ctx();
        // atomic + its own MULTI* -> GC (NOT flattened into a single MULTILINESTRING).
        assert_eq!(
            collect_wkts(
                &ctx,
                &[
                    "LINESTRING(0 0,1 1)",
                    "MULTILINESTRING((2 2,3 3),(4 4,5 5))"
                ]
            )
            .await
            .as_deref(),
            Some("GEOMETRYCOLLECTION(LINESTRING(0 0,1 1),MULTILINESTRING((2 2,3 3),(4 4,5 5)))")
        );
        // two MULTI* of the same kind -> GC of two multis (still not flattened).
        assert_eq!(
            collect_wkts(
                &ctx,
                &["MULTILINESTRING((0 0,1 1))", "MULTILINESTRING((2 2,3 3))"]
            )
            .await
            .as_deref(),
            Some("GEOMETRYCOLLECTION(MULTILINESTRING((0 0,1 1)),MULTILINESTRING((2 2,3 3)))")
        );
        // atomic + a MULTI* of another kind -> GC.
        assert_eq!(
            collect_wkts(&ctx, &["POINT(0 0)", "MULTIPOINT((1 1),(2 2))"])
                .await
                .as_deref(),
            Some("GEOMETRYCOLLECTION(POINT(0 0),MULTIPOINT((1 1),(2 2)))")
        );
    }

    /// Empty-member parity with PostGIS 3.6.3 / GEOS 3.13.1. geoarrow renders an empty member as
    /// `()` where PostGIS renders `EMPTY` (PostGIS: `MULTILINESTRING(EMPTY,EMPTY)`) — same geometry,
    /// different spelling.
    ///
    ///   SELECT ST_AsText(ST_Collect(ST_GeomFromText(w)))
    ///   FROM (VALUES ('LINESTRING EMPTY'),('LINESTRING EMPTY')) AS t(w);
    #[tokio::test]
    async fn test_collect_parity_empty_members() {
        let ctx = ctx();
        // Homogeneous empties still promote to the MULTI* (not NULL); each empty is kept.
        assert_eq!(
            collect_wkts(&ctx, &["LINESTRING EMPTY", "LINESTRING EMPTY"])
                .await
                .as_deref(),
            Some("MULTILINESTRING((),())") // PostGIS: MULTILINESTRING(EMPTY,EMPTY)
        );
        // Mixed bases (empty polygon + line) -> GC, the empty preserved.
        assert_eq!(
            collect_wkts(&ctx, &["POLYGON EMPTY", "LINESTRING(0 0,1 1)"])
                .await
                .as_deref(),
            Some("GEOMETRYCOLLECTION(POLYGON EMPTY,LINESTRING(0 0,1 1))")
        );
    }

    /// Higher-dimension parity with PostGIS 3.6.3 / GEOS 3.13.1: Z and M are both preserved.
    /// geoarrow renders the dimension tag tight (`MULTIPOINT ZM((...))`) where PostGIS spaces or
    /// elides it (`MULTIPOINT ZM (...)`, `MULTIPOINTM(...)`); the geometry is identical.
    ///
    ///   SELECT ST_AsEWKT(ST_Collect(ST_GeomFromText(w)))
    ///   FROM (VALUES ('POINT ZM(0 0 1 5)'),('POINT ZM(1 1 2 6)')) AS t(w);
    #[tokio::test]
    async fn test_collect_parity_zm_dimensions() {
        let ctx = ctx();
        assert_eq!(
            collect_wkts(&ctx, &["POINT ZM(0 0 1 5)", "POINT ZM(1 1 2 6)"])
                .await
                .as_deref(),
            Some("MULTIPOINT ZM((0 0 1 5),(1 1 2 6))")
        );
        assert_eq!(
            collect_wkts(&ctx, &["POINT M(0 0 5)", "POINT M(1 1 6)"])
                .await
                .as_deref(),
            Some("MULTIPOINT M((0 0 5),(1 1 6))")
        );
    }

    /// Multi-batch, multi-partition `GROUP BY` forces partial aggregation + merge through the
    /// `GroupsAccumulator` (`update_batch`/`state`/`merge_batch`/`evaluate`). Member order is not
    /// guaranteed across partitions, so assert membership rather than exact order.
    #[tokio::test]
    async fn test_collect_group_by_two_phase() {
        let ctx = ctx_two_phase();
        let df = ctx
            .sql(
                "SELECT k, ST_AsText(ST_Collect(ST_GeomFromText(w))) AS g FROM (VALUES \
                 (1,'POINT(0 0)'),(2,'POINT(5 5)'),(1,'POINT(1 1)'),(2,'POINT(6 6)'),(1,'POINT(2 2)') \
                 ) AS t(k,w) GROUP BY k ORDER BY k",
            )
            .await
            .unwrap();
        // `batch_size(1)` splits the output into one-row batches; gather them in (ORDER BY k) order.
        let batches = df.collect().await.unwrap();
        let rows: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let g = b.column_by_name("g").unwrap().as_string::<i32>();
                (0..b.num_rows())
                    .map(|i| g.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(rows.len(), 2);
        for (row, pts) in [
            (0usize, ["(0 0)", "(1 1)", "(2 2)"].as_slice()),
            (1usize, ["(5 5)", "(6 6)"].as_slice()),
        ] {
            let v = &rows[row];
            assert!(v.starts_with("MULTIPOINT"), "row {row}: {v}");
            assert_eq!(v.matches('(').count(), pts.len() + 1, "row {row}: {v}");
            for p in pts {
                assert!(v.contains(p), "row {row}: {v} missing {p}");
            }
        }
    }

    /// One query mixing the three group outcomes: homogeneous → MULTI*, mixed → GC, all-NULL → NULL.
    #[tokio::test]
    async fn test_collect_group_by_heterogeneous() {
        let ctx = ctx();
        let df = ctx
            .sql(
                "SELECT k, ST_AsText(ST_Collect(ST_GeomFromText(w))) AS g FROM (VALUES \
                 (1,'POINT(0 0)'),(1,'POINT(1 1)'), \
                 (2,'POINT(2 2)'),(2,'LINESTRING(3 3,4 4)'), \
                 (3,CAST(NULL AS TEXT)) \
                 ) AS t(k,w) GROUP BY k ORDER BY k",
            )
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let g = batch.column_by_name("g").unwrap().as_string::<i32>();
        assert_eq!(g.value(0), "MULTIPOINT((0 0),(1 1))");
        assert_eq!(
            g.value(1),
            "GEOMETRYCOLLECTION(POINT(2 2),LINESTRING(3 3,4 4))"
        );
        assert!(g.is_null(2), "all-NULL group should be NULL");
    }

    /// Ungrouped collect over many small batches/partitions exercises the rewritten simple
    /// `Accumulator` (`update`/`state`/`merge`/`evaluate`). Order is not guaranteed; assert membership.
    #[tokio::test]
    async fn test_collect_ungrouped_multibatch() {
        let ctx = ctx_two_phase();
        let v = text1(
            &ctx,
            "SELECT ST_AsText(ST_Collect(ST_GeomFromText(w))) FROM (VALUES \
             ('POINT(0 0)'),('POINT(1 1)'),('POINT(2 2)'),('POINT(3 3)')) AS t(w)",
        )
        .await
        .unwrap();
        assert!(v.starts_with("MULTIPOINT"), "{v}");
        assert_eq!(v.matches('(').count(), 5, "{v}"); // 4 members + outer paren
        for p in ["(0 0)", "(1 1)", "(2 2)", "(3 3)"] {
            assert!(v.contains(p), "{v} missing {p}");
        }
    }

    /// The high-cardinality fast path: `convert_to_state` turns each row into a one-element list,
    /// which `merge_batch` then groups. Drives the accumulator directly so the path is covered
    /// regardless of DataFusion's runtime heuristics.
    #[tokio::test]
    async fn test_groups_accumulator_convert_to_state() {
        let ctx = ctx();
        let (array, field) = geom_array(
            &ctx,
            &[Some("POINT(0 0)"), Some("POINT(1 1)"), Some("POINT(2 2)")],
        )
        .await;
        let meta = collect_meta(&field);

        let acc = CollectGroupsAccumulator::new(field.clone(), meta.clone(), CoordType::default());
        let states = acc.convert_to_state(&[array], None).unwrap();

        // rows 0 & 2 -> group 0, row 1 -> group 1.
        let mut merged = CollectGroupsAccumulator::new(field, meta.clone(), CoordType::default());
        merged.merge_batch(&states, &[0, 1, 0], None, 2).unwrap();
        let out = merged.evaluate(EmitTo::All).unwrap();

        assert_eq!(
            geom_wkts(&out, meta, CoordType::default()),
            vec![
                "MULTIPOINT((0 0),(2 2))".to_string(),
                "MULTIPOINT((1 1))".to_string()
            ]
        );
    }

    /// `EmitTo::First` emits a prefix of groups and shifts the rest down — the spill path. Exercises
    /// `compact_retained_state` (drop/compact batches, renumber retained groups).
    #[tokio::test]
    async fn test_groups_accumulator_emit_first() {
        let ctx = ctx();
        let (array, field) = geom_array(
            &ctx,
            &[Some("POINT(0 0)"), Some("POINT(1 1)"), Some("POINT(2 2)")],
        )
        .await;
        let meta = collect_meta(&field);
        let mut acc =
            CollectGroupsAccumulator::new(field.clone(), meta.clone(), CoordType::default());
        acc.update_batch(&[array], &[0, 1, 2], None, 3).unwrap();

        // Emit group 0 only; groups 1,2 shift down to 0,1.
        let first = acc.evaluate(EmitTo::First(1)).unwrap();
        assert_eq!(
            geom_wkts(&first, meta.clone(), CoordType::default()),
            vec!["MULTIPOINT((0 0))".to_string()]
        );

        // Add a member to (shifted) group 1, originally group 2.
        let (array2, _) = geom_array(&ctx, &[Some("POINT(9 9)")]).await;
        acc.update_batch(&[array2], &[1], None, 2).unwrap();
        let rest = acc.evaluate(EmitTo::All).unwrap();
        assert_eq!(
            geom_wkts(&rest, meta, CoordType::default()),
            vec![
                "MULTIPOINT((1 1))".to_string(),
                "MULTIPOINT((2 2),(9 9))".to_string()
            ]
        );
    }

    /// NULL and filtered-out rows are dropped by `convert_to_state` (NULL lists), so they never reach
    /// a group; an all-NULL group emits NULL.
    #[tokio::test]
    async fn test_groups_accumulator_convert_to_state_nulls() {
        let ctx = ctx();
        let (array, field) =
            geom_array(&ctx, &[Some("POINT(0 0)"), None, Some("POINT(2 2)")]).await;
        let meta = collect_meta(&field);

        let acc = CollectGroupsAccumulator::new(field.clone(), meta.clone(), CoordType::default());
        let states = acc.convert_to_state(&[array], None).unwrap();

        // row 0 -> group 0, NULL row 1 -> group 1 (stays empty -> NULL), row 2 -> group 0.
        let mut merged = CollectGroupsAccumulator::new(field, meta.clone(), CoordType::default());
        merged.merge_batch(&states, &[0, 1, 0], None, 2).unwrap();
        let out = merged.evaluate(EmitTo::All).unwrap();

        assert_eq!(
            geom_wkts(&out, meta, CoordType::default()),
            vec!["MULTIPOINT((0 0),(2 2))".to_string(), "NULL".to_string()]
        );
    }
}

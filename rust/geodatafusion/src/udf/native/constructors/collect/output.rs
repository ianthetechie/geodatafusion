//! Final `ST_Collect` output assembly for both WKB-backed and native member stores.

use std::sync::Arc;

use arrow_array::ArrayRef;
use datafusion::error::DataFusionError;
use geoarrow_array::array::{GeometryArray, WkbArray};
use geoarrow_array::builder::{GeometryBuilder, WkbBuilder};
use geoarrow_array::cast::from_wkb;
use geoarrow_array::scalar::Geometry;
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::{CoordType, GeoArrowType, GeometryType, Metadata, WkbType};

use super::container::{CollectedGeometry, classify_members};
use super::state::{MemberBatches, MemberStorage};
use crate::error::{GeoDataFusionError, GeoDataFusionResult};

/// Assemble one output geometry per group from stored member batches.
pub(super) fn assemble_output(
    batches: &MemberBatches,
    groups: &[&[(usize, usize)]],
    metadata: Arc<Metadata>,
    coord_type: CoordType,
) -> GeoDataFusionResult<ArrayRef> {
    match batches.storage() {
        MemberStorage::Wkb(batches) => assemble_wkb(batches, groups, metadata, coord_type),
        MemberStorage::Geometry(batches) => {
            let typed: Vec<&GeometryArray> = batches.iter().map(|a| a.as_ref()).collect();
            assemble_geometry(&typed, groups, metadata, coord_type)
        }
    }
}

fn assemble_geometry<'a>(
    typed: &[&'a GeometryArray],
    groups: &[&[(usize, usize)]],
    metadata: Arc<Metadata>,
    coord_type: CoordType,
) -> GeoDataFusionResult<ArrayRef> {
    let out_type = GeometryType::new(metadata).with_coord_type(coord_type);
    let mut builder = GeometryBuilder::new(out_type);
    let mut members: Vec<Geometry<'a>> = Vec::new();
    for group in groups {
        if group.is_empty() {
            builder.push_null();
            continue;
        }
        members.clear();
        for &(bi, ri) in *group {
            if !typed[bi].is_null(ri) {
                members.push(typed[bi].value(ri)?);
            }
        }
        if members.is_empty() {
            builder.push_null();
            continue;
        }
        let (kind, dim) = classify_members(&members)?;
        builder.push_geometry(Some(&CollectedGeometry::new(&members, kind, dim)))?;
    }
    Ok(builder.finish().into_array_ref())
}

fn assemble_wkb(
    batches: &[Arc<WkbArray>],
    groups: &[&[(usize, usize)]],
    metadata: Arc<Metadata>,
    coord_type: CoordType,
) -> GeoDataFusionResult<ArrayRef> {
    let mut members: Vec<&[u8]> = Vec::new();
    let mut containers = Vec::with_capacity(groups.len());
    for group in groups {
        members.clear();
        for &(bi, ri) in *group {
            if !batches[bi].is_null(ri) {
                members.push(batches[bi].inner().value(ri));
            }
        }
        containers.push(if members.is_empty() {
            None
        } else {
            Some(build_container_wkb(&members)?)
        });
    }
    build_geometry_array(containers, metadata, coord_type)
}

/// Decode assembled container WKB into the mixed `Geometry` output type.
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


const WKB_NDR: u8 = 1;

// ISO-WKB type codes are `base + dimension_offset`; matching dimensions must share the same offset.
const DIM_MODULUS: u32 = 1000;
const MAX_ATOMIC_BASE: u32 = 3;
const ATOMIC_TO_MULTI: u32 = 3;
const GEOMETRY_COLLECTION_BASE: u32 = 7;

// EWKB encodes Z/M/SRID via these high bits of the type code; ISO WKB never sets them.
const EWKB_FLAGS: u32 = 0x8000_0000 | 0x4000_0000 | 0x2000_0000;

fn unsupported_wkb_encoding() -> GeoDataFusionError {
    DataFusionError::NotImplemented(
        "ST_Collect requires ISO little-endian WKB; EWKB or big-endian WKB input is not supported"
            .to_string(),
    )
    .into()
}

/// Read a member's ISO-WKB type code, rejecting anything that is not ISO little-endian.
///
/// The WKB aggregate-state path concatenates member bytes into one MULTI*/GEOMETRYCOLLECTION
/// container and hands it to geoarrow's reader. That reader's MULTI* decoders fixed-stride and
/// assume every member shares the container's little-endian byte order and dimension and carries
/// no SRID (upstream `wkb` crate `multipoint.rs`/`multilinestring.rs`), so heterogeneously-encoded
/// members cannot be embedded in a MULTI* at all. geoarrow's own encoder always emits ISO
/// little-endian WKB, so the only way to reach this code with EWKB/big-endian bytes is an
/// externally-supplied `geoarrow.wkb` column (stored verbatim, not re-encoded on read).
///
/// This function ensures we don't silently emit a corrupt container.
/// Supporting EWKB would require re-encoding members to ISO at this point.
fn iso_member_type_code(wkb: &[u8]) -> GeoDataFusionResult<u32> {
    let header: [u8; 4] = wkb
        .get(1..5)
        .and_then(|h| h.try_into().ok())
        .filter(|_| wkb[0] == WKB_NDR)
        .ok_or_else(unsupported_wkb_encoding)?;
    let code = u32::from_le_bytes(header);
    if code & EWKB_FLAGS != 0 {
        return Err(unsupported_wkb_encoding());
    }
    Ok(code)
}

/// Wrap member WKB values in the MULTI* or GEOMETRYCOLLECTION container required by `ST_Collect`.
fn build_container_wkb(members: &[&[u8]]) -> GeoDataFusionResult<Vec<u8>> {
    // Validate and read every member's ISO type code once.
    let codes: Vec<u32> = members
        .iter()
        .map(|m| iso_member_type_code(m))
        .collect::<GeoDataFusionResult<_>>()?;

    let code0 = codes[0];
    let (base0, dim0) = (code0 % DIM_MODULUS, code0 - code0 % DIM_MODULUS);

    if codes.iter().any(|&code| code - code % DIM_MODULUS != dim0) {
        return Err(DataFusionError::NotImplemented(
            "ST_Collect cannot combine geometries of differing coordinate dimensions (e.g. XY and \
             XYZ)"
                .to_string(),
        )
        .into());
    }

    let homogeneous_atomic = (1..=MAX_ATOMIC_BASE).contains(&base0)
        && codes.iter().all(|&code| code % DIM_MODULUS == base0);

    let container_base = if homogeneous_atomic {
        base0 + ATOMIC_TO_MULTI
    } else {
        GEOMETRY_COLLECTION_BASE
    };
    let container_code = container_base + dim0;

    let total: usize = members.iter().map(|m| m.len()).sum();
    let mut buf = Vec::with_capacity(9 + total);
    buf.push(WKB_NDR);
    buf.extend_from_slice(&container_code.to_le_bytes());
    buf.extend_from_slice(&(members.len() as u32).to_le_bytes());
    for m in members {
        buf.extend_from_slice(m);
    }
    Ok(buf)
}

#[cfg(test)]
mod test {
    use super::*;

    fn iso_le_point() -> Vec<u8> {
        // ISO little-endian `POINT(0 0)`: byte order `01`, type `1`, then x/y.
        let mut b = vec![WKB_NDR];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b
    }

    fn ewkb_srid_point() -> Vec<u8> {
        // EWKB `POINT(0 0)` with the SRID flag set (type `0x20000001`, SRID 4326).
        let mut b = vec![WKB_NDR];
        b.extend_from_slice(&0x2000_0001u32.to_le_bytes());
        b.extend_from_slice(&4326u32.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b
    }

    #[test]
    fn iso_member_type_code_reads_iso_le() {
        assert_eq!(iso_member_type_code(&iso_le_point()).unwrap(), 1);
    }

    #[test]
    fn iso_member_type_code_rejects_ewkb_srid() {
        assert!(iso_member_type_code(&ewkb_srid_point()).is_err());
    }

    #[test]
    fn iso_member_type_code_rejects_ewkb_z() {
        // EWKB PointZ (type 0x80000001).
        let mut b = vec![WKB_NDR];
        b.extend_from_slice(&0x8000_0001u32.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        b.extend_from_slice(&0f64.to_le_bytes());
        assert!(iso_member_type_code(&b).is_err());
    }

    #[test]
    fn iso_member_type_code_rejects_big_endian() {
        // Big-endian (XDR) ISO Point: byte order 00, big-endian type code.
        let mut b = vec![0u8];
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&0f64.to_be_bytes());
        b.extend_from_slice(&0f64.to_be_bytes());
        assert!(iso_member_type_code(&b).is_err());
    }

    #[test]
    fn iso_member_type_code_rejects_truncated() {
        assert!(iso_member_type_code(&[WKB_NDR, 0x01]).is_err());
    }

    #[test]
    fn build_container_wkb_rejects_ewkb_member() {
        let iso = iso_le_point();
        let ewkb = ewkb_srid_point();
        let members: Vec<&[u8]> = vec![iso.as_slice(), ewkb.as_slice()];
        let err = build_container_wkb(&members).unwrap_err();
        assert!(
            err.to_string().contains("ISO little-endian"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn build_container_wkb_builds_iso_multipoint() {
        let p0 = iso_le_point();
        let p1 = iso_le_point();
        let members: Vec<&[u8]> = vec![&p0, &p1];
        let buf = build_container_wkb(&members).unwrap();
        assert_eq!(buf[0], WKB_NDR);
        // MultiPoint (4), two members.
        assert_eq!(u32::from_le_bytes(buf[1..5].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(buf[5..9].try_into().unwrap()), 2);
    }
}

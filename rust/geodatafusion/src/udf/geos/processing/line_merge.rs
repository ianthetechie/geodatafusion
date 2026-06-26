use std::any::Any;
use std::sync::{Arc, LazyLock, OnceLock};

use arrow_schema::{DataType, FieldRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion::scalar::ScalarValue;
use geoarrow_array::GeoArrowArray;
use geoarrow_array::array::{GeometryArray, from_arrow_array};
use geoarrow_expr_geos::export::array::ToGEOS;
use geoarrow_expr_geos::import::array::FromGEOS;
use geoarrow_schema::{CoordType, GeoArrowType, GeometryType, Metadata};
use geos::Geom;

use crate::data_types::any_geometry_type;
use crate::error::GeoDataFusionResult;

/// A single geometry argument, optionally followed by the `directed` boolean.
static SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    let geometry_types = any_geometry_type();
    let mut variants = Vec::with_capacity(geometry_types.len() * 2);
    for geometry_type in geometry_types {
        variants.push(TypeSignature::Exact(vec![geometry_type.clone()]));
        variants.push(TypeSignature::Exact(vec![geometry_type, DataType::Boolean]));
    }
    Signature::one_of(variants, Volatility::Immutable)
});

/// Sews together the component lines of a (multi)linestring.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct LineMerge {
    coord_type: CoordType,
}

impl LineMerge {
    pub fn new(coord_type: CoordType) -> Self {
        Self { coord_type }
    }
}

impl Default for LineMerge {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

impl ScalarUDFImpl for LineMerge {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_linemerge"
    }

    fn signature(&self) -> &Signature {
        &SIGNATURE
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(DataFusionError::Internal("return_type".to_string()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(return_field_impl(args, self.coord_type)?)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(line_merge_impl(args)?)
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Returns a (set of) LineString(s) formed by sewing together the constituent line work of a MultiLineString. Lines are joined at endpoints where exactly two lines meet; lines are not merged across intersections of three or more lines. When `directed` is true, lines are only merged when their directions agree. Non-linear inputs yield an empty GeometryCollection. This function strips the M dimension.",
                "ST_LineMerge(geometry[, directed])",
            )
            .with_argument("geom", "geometry")
            .with_argument("directed", "boolean")
            .build()
        }))
    }
}

fn return_field_impl(
    args: ReturnFieldArgs,
    coord_type: CoordType,
) -> GeoDataFusionResult<FieldRef> {
    let metadata = Arc::new(Metadata::try_from(args.arg_fields[0].as_ref()).unwrap_or_default());
    let output_type = GeometryType::new(metadata).with_coord_type(coord_type);
    Ok(Arc::new(output_type.to_field("", true)))
}

/// Parse the optional `directed` argument.
///
/// Absent or null is treated as `false`.
fn parse_directed(args: &ScalarFunctionArgs) -> GeoDataFusionResult<bool> {
    match args.args.get(1) {
        None => Ok(false),
        Some(arg) => match arg.cast_to(&DataType::Boolean, None)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(directed)) => Ok(directed.unwrap_or(false)),
            // A cast to `Boolean` only ever yields a `Boolean` scalar.
            ColumnarValue::Scalar(_) => unreachable!("cast to Boolean yields a Boolean scalar"),
            ColumnarValue::Array(_) => Err(DataFusionError::NotImplemented(
                "Vectorized `directed` argument to ST_LineMerge is not yet implemented".to_string(),
            )
            .into()),
        },
    }
}

fn line_merge_impl(args: ScalarFunctionArgs) -> GeoDataFusionResult<ColumnarValue> {
    // Parse the directed argument
    let directed = parse_directed(&args)?;

    let arrays = ColumnarValue::values_to_arrays(&args.args[0..1])?;
    let geo_array = from_arrow_array(&arrays[0], &args.arg_fields[0])?;

    // Convert the array to GEOS geometries, merge each, then convert back. GEOS has no concept of
    // an M dimension, so M is dropped on the way in (matching PostGIS, which strips M through
    // ST_LineMerge); Z is preserved end-to-end.
    let merged = geo_array
        .as_ref()
        .to_geos()?
        .into_iter()
        .map(|maybe_geom| {
            // Null inputs propagate to null outputs.
            maybe_geom
                .map(|geom| {
                    if geom.is_empty()? {
                        // PostGIS returns the original geometry for empty input, whereas GEOS
                        // would collapse it to an empty GeometryCollection. Preserve the PostGIS
                        // behavior. Thanks to Dewy for pointing out from the SedonaDB implementation:
                        // https://github.com/apache/sedona-db/blob/cf8b9ceaf7a78c042bf73ab0e5040187046fe256/c/sedona-geos/src/st_line_merge.rs#L105-L131!
                        Ok(geom)
                    } else if directed {
                        geom.line_merge_directed()
                    } else {
                        geom.line_merge()
                    }
                })
                .transpose()
        })
        .collect::<std::result::Result<Vec<_>, geos::Error>>()?;

    // Convert the merged GEOS geometries back into a GeoArrow `GeometryArray`.
    let to_type = GeoArrowType::from_arrow_field(args.return_field.as_ref())?;
    let GeoArrowType::Geometry(geometry_type) = to_type else {
        return Err(DataFusionError::Internal(
            "ST_LineMerge expected a Geometry return type".to_string(),
        )
        .into());
    };
    let result = GeometryArray::from_geos(merged, geometry_type)?;

    Ok(ColumnarValue::Array(result.to_array_ref()))
}

#[cfg(test)]
mod test {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::udf::native::constructors::CollectAggregate;
    use crate::udf::native::io::{AsText, GeomFromText, GeomFromWKB};

    fn ctx() -> SessionContext {
        let ctx = SessionContext::new();
        ctx.register_udf(LineMerge::default().into());
        ctx.register_udf(GeomFromText::default().into());
        ctx.register_udf(GeomFromWKB::new(Default::default()).into());
        ctx.register_udf(AsText.into());
        // Registered for the ST_LineMerge(ST_Collect(...)) composition parity test.
        ctx.register_udaf(CollectAggregate::default().into());
        ctx
    }

    #[tokio::test]
    async fn test_st_linemerge() {
        let ctx = ctx();

        // Explicitly noted examples come from the PostGIS documentation (CC-BY-SA-3.0).
        let cases = vec![
            (
                "MULTILINESTRING((10 160, 60 120), (120 140, 60 120), (120 140, 180 120))",
                "LINESTRING(10 160,60 120,120 140,180 120)",
                "PostGIS doc example: lines meeting two-at-a-time sew into one LineString",
            ),
            (
                "MULTILINESTRING((10 160, 60 120), (120 140, 60 120), (120 140, 180 120), (100 180, 120 140))",
                "MULTILINESTRING((10 160,60 120,120 140),(100 180,120 140),(120 140,180 120))",
                "degree-3 node: merge does not cross a junction of three lines",
            ),
            (
                "MULTILINESTRING((-29 -27,-30 -29.7,-36 -31,-45 -33),(-45.2 -33.2,-46 -32))",
                "MULTILINESTRING((-45.2 -33.2,-46 -32),(-29 -27,-30 -29.7,-36 -31,-45 -33))",
                "disjoint components are returned unchanged (as a MultiLineString)",
            ),
            (
                "LINESTRING(0 0, 1 1)",
                "LINESTRING(0 0,1 1)",
                "a lone LineString round-trips unchanged",
            ),
            (
                "POINT(0 0)",
                "GEOMETRYCOLLECTION EMPTY",
                "input with no line work yields an empty GeometryCollection",
            ),
            (
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "LINESTRING(0 0,1 0,1 1,0 1,0 0)",
                "PostGIS gotcha: polygons pass through to GEOS unfiltered, so the boundary ring is returned as a closed LineString rather than an empty GeometryCollection",
            ),
            (
                "LINESTRING EMPTY",
                "LINESTRING EMPTY",
                "PostGIS gotcha: empty input is returned as-is rather than collapsed to an empty GeometryCollection",
            ),
            (
                "MULTILINESTRING Z((-29 -27 11,-30 -29.7 10,-36 -31 5,-45 -33 6), (-29 -27 12,-30 -29.7 5), (-45 -33 1,-46 -32 11))",
                "LINESTRING Z(-30 -29.7 5,-29 -27 11,-30 -29.7 10,-36 -31 5,-45 -33 1,-46 -32 11)",
                "PostGIS example with Z-dimension handling",
            ),
        ];

        for (input, expected, description) in cases {
            let sql = format!(
                "SELECT ST_AsText(ST_LineMerge(ST_GeomFromText('{}')))",
                input
            );
            let df = ctx
                .sql(&sql)
                .await
                .unwrap_or_else(|_| panic!("Failed to execute SQL for {}", description));

            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let val = batch.column(0).as_string::<i32>().value(0);

            assert_eq!(val, expected, "Failed on {}: {}", description, input);
        }
    }

    #[tokio::test]
    async fn test_st_linemerge_directed() {
        let ctx = ctx();

        // Same input, contrasting the directed flag: with TRUE the disagreeing segments stay
        // split; with FALSE (the default) they all merge into a single LineString.
        let input = "MULTILINESTRING((60 30, 10 70), (120 50, 60 30), (120 50, 180 30))";
        let cases = vec![
            (
                "TRUE",
                "MULTILINESTRING((120 50,60 30,10 70),(120 50,180 30))",
                "directed: segments whose directions disagree are not merged",
            ),
            (
                "FALSE",
                "LINESTRING(180 30,120 50,60 30,10 70)",
                "undirected: same input merges fully when direction is ignored",
            ),
        ];

        for (directed, expected, description) in cases {
            let sql = format!(
                "SELECT ST_AsText(ST_LineMerge(ST_GeomFromText('{}'), {}))",
                input, directed
            );
            let df = ctx
                .sql(&sql)
                .await
                .unwrap_or_else(|_| panic!("Failed to execute SQL for {description}"));

            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let val = batch.column(0).as_string::<i32>().value(0);

            assert_eq!(
                val, expected,
                "Failed on {description}: directed={directed}",
            );
        }
    }

    #[tokio::test]
    async fn test_st_linemerge_null() {
        let ctx = ctx();

        // Null geometries propagate to null, including alongside non-null rows in the same array.
        let df = ctx
            .sql(
                "WITH t(wkt) AS (VALUES ('LINESTRING(0 0, 1 1)'), (CAST(NULL AS TEXT))) \
                 SELECT ST_AsText(ST_LineMerge(ST_GeomFromText(wkt))) FROM t ORDER BY wkt NULLS LAST",
            )
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let col = batch.column(0).as_string::<i32>();

        assert_eq!(col.value(0), "LINESTRING(0 0,1 1)");
        assert!(
            col.is_null(1),
            "null input should propagate to a null output"
        );
    }

    #[tokio::test]
    async fn test_st_linemerge_compound_and_dim_parity() {
        let ctx = ctx();

        let mut cases: Vec<(&str, &str, &str)> = vec![
            (
                "GEOMETRYCOLLECTION(LINESTRING(0 0,1 1),LINESTRING(1 1,2 2))",
                "LINESTRING(0 0,1 1,2 2)",
                "GEOS extracts and merges the linework recursively out of a GeometryCollection",
            ),
            (
                "GEOMETRYCOLLECTION(POINT(9 9),LINESTRING(0 0,1 1),LINESTRING(1 1,2 2),POLYGON((5 5,6 5,6 6,5 6,5 5)))",
                "MULTILINESTRING((0 0,1 1,2 2),(5 5,6 5,6 6,5 6,5 5))",
                "in a mixed GC the point is dropped while polygon rings join the merged linework",
            ),
            (
                "POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,3 2,3 3,2 3,2 2))",
                "MULTILINESTRING((0 0,10 0,10 10,0 10,0 0),(2 2,3 2,3 3,2 3,2 2))",
                "a polygon's shell and holes come back as their boundary rings",
            ),
            (
                "MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((5 5,6 5,6 6,5 6,5 5)))",
                "MULTILINESTRING((0 0,1 0,1 1,0 1,0 0),(5 5,6 5,6 6,5 6,5 5))",
                "every ring of every polygon is returned",
            ),
            (
                "LINESTRING(0 0,1 0,1 1,0 1,0 0)",
                "LINESTRING(0 0,1 0,1 1,0 1,0 0)",
                "a closed ring is a valid lone LineString and round-trips unchanged",
            ),
            (
                "MULTIPOINT((0 0),(1 1))",
                "GEOMETRYCOLLECTION EMPTY",
                "no linework yields an empty GeometryCollection",
            ),
            (
                "MULTILINESTRING Z((0 0 1,1 1 2),(1 1 2,2 2 3))",
                "LINESTRING Z(0 0 1,1 1 2,2 2 3)",
                "Z is preserved through the merge (PostGIS: 'LINESTRING Z (0 0 1,1 1 2,2 2 3)')",
            ),
        ];

        // M-bearing inputs can only be converted into GEOS when built against GEOS 3.14+
        // (GEOS has no M concept before then). GEOS still strips M through the merge, matching
        // PostGIS, so the expected outputs carry no M.
        if cfg!(feature = "geos-3_14") {
            cases.extend([
                (
                    "MULTILINESTRING M((0 0 1,1 1 2),(1 1 2,2 2 3))",
                    "LINESTRING(0 0,1 1,2 2)",
                    "M is stripped (GEOS has no M dimension), matching PostGIS",
                ),
                (
                    "MULTILINESTRING ZM((0 0 1 7,1 1 2 8),(1 1 2 8,2 2 3 9))",
                    "LINESTRING Z(0 0 1,1 1 2,2 2 3)",
                    "Z kept, M stripped (PostGIS: 'LINESTRING Z (0 0 1,1 1 2,2 2 3)')",
                ),
            ]);
        }

        for (input, expected, description) in cases {
            let sql = format!("SELECT ST_AsText(ST_LineMerge(ST_GeomFromText('{input}')))");
            let df = ctx
                .sql(&sql)
                .await
                .unwrap_or_else(|_| panic!("Failed to execute SQL for {description}"));
            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let val = batch.column(0).as_string::<i32>().value(0);
            assert_eq!(val, expected, "Failed on {description}: {input}");
        }
    }

    /// Empty multi-geometry parity. geoarrow's WKT parser crashes on empty multis, so the input is
    /// supplied as ISO-WKB hex (`MULTILINESTRING EMPTY`), as the empty-geometry fixtures elsewhere do.
    ///
    /// PostGIS (verify):
    ///   SELECT ST_AsText(ST_LineMerge(ST_GeomFromWKB(decode('010500000000000000','hex'))));
    ///   -- MULTILINESTRING EMPTY   (empty input is returned as-is, not collapsed to GC EMPTY)
    #[tokio::test]
    async fn test_st_linemerge_empty_multi_parity() {
        let ctx = ctx();
        let df = ctx
            .sql("SELECT ST_AsText(ST_LineMerge(ST_GeomFromWKB(X'010500000000000000')))")
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let val = batch.column(0).as_string::<i32>().value(0);
        assert_eq!(val, "MULTILINESTRING EMPTY");
    }

    /// `ST_LineMerge(ST_Collect(...))` parity with PostGIS 3.6.3 / GEOS 3.13.1 — collect the rows of
    /// a VALUES list, then merge the resulting (Multi)LineString / GeometryCollection.
    ///
    /// Verify any row by pasting into psql ("postgresql://osm:osm@localhost:5432/osm"), e.g.:
    ///   SELECT ST_AsText(ST_LineMerge(ST_Collect(ST_GeomFromText(w))))
    ///   FROM (VALUES ('LINESTRING(0 0,1 1)'),('MULTILINESTRING((1 1,2 2),(2 2,3 3))')) AS t(w);
    #[tokio::test]
    async fn test_st_linemerge_of_collect_parity() {
        let ctx = ctx();

        // (collected members, expected merged WKT, description)
        let cases: Vec<(&[&str], &str, &str)> = vec![
            (
                &["LINESTRING(0 0,1 1)", "LINESTRING(1 1,2 2)"],
                "LINESTRING(0 0,1 1,2 2)",
                "Collect -> MULTILINESTRING, then merged end-to-end",
            ),
            (
                &[
                    "LINESTRING(0 0,1 1)",
                    "MULTILINESTRING((1 1,2 2),(2 2,3 3))",
                ],
                "LINESTRING(0 0,1 1,2 2,3 3)",
                "Collect -> GEOMETRYCOLLECTION (one multi member), merged recursively all the same",
            ),
            (
                &[
                    "LINESTRING(0 0,1 1)",
                    "LINESTRING(1 1,2 2)",
                    "LINESTRING EMPTY",
                ],
                "LINESTRING(0 0,1 1,2 2)",
                "an empty member does not obstruct the merge",
            ),
            (
                &[
                    "LINESTRING(0 0,1 1)",
                    "LINESTRING(1 1,2 2)",
                    "POLYGON((5 5,6 5,6 6,5 6,5 5))",
                ],
                "MULTILINESTRING((0 0,1 1,2 2),(5 5,6 5,6 6,5 6,5 5))",
                "a collected polygon contributes its boundary ring to the merge",
            ),
            (
                &["LINESTRING(0 0,1 1)", "LINESTRING(1 1,2 2)", "POINT(9 9)"],
                "LINESTRING(0 0,1 1,2 2)",
                "a collected point is dropped",
            ),
            (
                &["LINESTRING Z(0 0 1,1 1 2)", "LINESTRING Z(1 1 2,2 2 3)"],
                "LINESTRING Z(0 0 1,1 1 2,2 2 3)",
                "Z survives Collect and merge (PostGIS: 'LINESTRING Z (0 0 1,1 1 2,2 2 3)')",
            ),
            (
                &["POINT(0 0)", "POINT(1 1)"],
                "GEOMETRYCOLLECTION EMPTY",
                "Collect -> MULTIPOINT has no linework, so the merge is empty",
            ),
        ];

        for (members, expected, description) in cases {
            let values = members
                .iter()
                .map(|w| format!("('{w}')"))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT ST_AsText(ST_LineMerge(ST_CollectAgg(ST_GeomFromText(w)))) \
                 FROM (VALUES {values}) AS t(w)"
            );
            let df = ctx
                .sql(&sql)
                .await
                .unwrap_or_else(|_| panic!("Failed to execute SQL for {description}"));
            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let val = batch.column(0).as_string::<i32>().value(0);
            assert_eq!(val, expected, "Failed on {description}: {members:?}");
        }
    }
}

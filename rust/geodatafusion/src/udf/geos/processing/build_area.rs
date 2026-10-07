use std::any::Any;
use std::sync::{Arc, OnceLock};

use arrow_schema::{DataType, FieldRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};
use geoarrow_array::GeoArrowArray;
use geoarrow_array::array::{GeometryArray, from_arrow_array};
use geoarrow_expr_geos::export::array::ToGEOS;
use geoarrow_expr_geos::import::array::FromGEOS;
use geoarrow_schema::{CoordType, GeoArrowType, GeometryType, Metadata};
use geos::{Geom, Geometry};

use crate::data_types::any_single_geometry_type_input;
use crate::error::GeoDataFusionResult;

/// Creates polygonal geometry from the linework of a geometry.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct BuildArea {
    coord_type: CoordType,
}

impl BuildArea {
    pub fn new(coord_type: CoordType) -> Self {
        Self { coord_type }
    }
}

impl Default for BuildArea {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

impl ScalarUDFImpl for BuildArea {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_buildarea"
    }

    fn signature(&self) -> &Signature {
        any_single_geometry_type_input()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(DataFusionError::Internal("return_type".to_string()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(return_field_impl(args, self.coord_type)?)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(build_area_impl(args)?)
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Creates polygonal geometry from the constituent linework of the input geometry. Inner rings become holes. If non-empty input linework does not form polygons, the result is NULL. Input linework must be correctly noded; crossing linework can produce invalid polygons. This function preserves Z and follows the GEOS bridge's M-dimension support.",
                "ST_BuildArea(geometry)",
            )
            .with_argument("geom", "geometry")
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

fn build_area_impl(args: ScalarFunctionArgs) -> GeoDataFusionResult<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let geo_array = from_arrow_array(&arrays[0], &args.arg_fields[0])?;

    // GEOS has no M support before 3.14; the existing conversion bridge controls how Z/M
    // coordinates are passed through, consistently with the other GEOS-backed UDFs.
    let areas = geo_array
        .as_ref()
        .to_geos()?
        .into_iter()
        .map(|maybe_geom| match maybe_geom {
            None => Ok(None),
            Some(geom) if geom.is_empty()? => Geometry::create_empty_polygon().map(Some),
            Some(geom) => {
                let area = geom.build_area()?;
                if area.is_empty()? {
                    // PostGIS returns NULL for non-empty geometries with no polygonal linework.
                    Ok(None)
                } else {
                    Ok(Some(area))
                }
            }
        })
        .collect::<std::result::Result<Vec<_>, geos::Error>>()?;

    let to_type = GeoArrowType::from_arrow_field(args.return_field.as_ref())?;
    let GeoArrowType::Geometry(geometry_type) = to_type else {
        return Err(DataFusionError::Internal(
            "ST_BuildArea expected a Geometry return type".to_string(),
        )
        .into());
    };
    let result = GeometryArray::from_geos(areas, geometry_type)?;

    Ok(ColumnarValue::Array(result.to_array_ref()))
}

#[cfg(test)]
mod test {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::udf::native::io::{AsText, GeomFromText};

    fn ctx() -> SessionContext {
        let ctx = SessionContext::new();
        ctx.register_udf(BuildArea::default().into());
        ctx.register_udf(GeomFromText::default().into());
        ctx.register_udf(AsText.into());
        ctx
    }

    #[tokio::test]
    async fn builds_areas_from_linework() {
        let ctx = ctx();

        // This example's input linework and expected output are from the PostGIS documentation
        // (CC-BY-SA-3.0): https://postgis.net/docs/ST_BuildArea.html
        let doc_input = "MULTILINESTRING((180 40,30 20,20 90),(180 40,160 160),(160 160,80 190,80 120,20 90),(80 60,120 130,150 80),(80 60,150 80))";
        let doc_expected = "POLYGON((180 40,30 20,20 90,80 120,80 190,160 160,180 40),(150 80,120 130,80 60,150 80))";

        let cases = [
            (
                doc_input,
                doc_expected,
                "PostGIS documentation linework forms a polygon with a hole",
            ),
            (
                "POLYGON((0 0,10 0,10 10,0 10,0 0))",
                "POLYGON((0 0,0 10,10 10,10 0,0 0))",
                "a polygon input is rebuilt from its boundary",
            ),
            (
                "GEOMETRYCOLLECTION(POINT(9 9),LINESTRING(0 0,1 0,1 1,0 1,0 0))",
                "POLYGON((0 0,0 1,1 1,1 0,0 0))",
                "non-linework members of a GeometryCollection are ignored",
            ),
        ];

        for (input, expected, description) in cases {
            let sql = format!("SELECT ST_AsText(ST_BuildArea(ST_GeomFromText('{input}')))");
            let df = ctx
                .sql(&sql)
                .await
                .unwrap_or_else(|_| panic!("Failed to execute SQL for {description}"));
            let batch = df.collect().await.unwrap().into_iter().next().unwrap();
            let val = batch.column(0).as_string::<i32>().value(0);

            assert_eq!(val, expected, "Failed on {description}: {input}");
        }
    }

    #[tokio::test]
    async fn builds_multipolygons_and_preserves_z() {
        let ctx = ctx();

        let multi = ctx
            .sql("SELECT ST_AsText(ST_BuildArea(ST_GeomFromText('MULTILINESTRING((0 0,2 0,2 2,0 2,0 0),(5 5,6 5,6 6,5 6,5 5))'))) ")
            .await
            .unwrap();
        let batch = multi.collect().await.unwrap().into_iter().next().unwrap();
        let value = batch.column(0).as_string::<i32>().value(0);
        assert!(
            value.starts_with("MULTIPOLYGON("),
            "expected a MultiPolygon, got {value}"
        );

        let with_z = ctx
            .sql("SELECT ST_AsText(ST_BuildArea(ST_GeomFromText('LINESTRING Z(0 0 7,10 0 7,10 10 7,0 10 7,0 0 7)'))) ")
            .await
            .unwrap();
        let batch = with_z.collect().await.unwrap().into_iter().next().unwrap();
        let value = batch.column(0).as_string::<i32>().value(0);
        assert!(
            value.starts_with("POLYGON Z("),
            "expected Z to be preserved, got {value}"
        );
        assert!(
            value.contains("0 0 7"),
            "expected Z coordinates, got {value}"
        );
    }

    #[tokio::test]
    async fn handles_null_non_area_and_empty_inputs() {
        let ctx = ctx();
        let df = ctx
            .sql(
                "WITH t(id, wkt) AS (VALUES \
                 (1, 'LINESTRING(0 0,1 0,1 1,0 1,0 0)'), \
                 (2, 'LINESTRING(0 0,1 1)'), \
                 (3, 'POINT(0 0)'), \
                 (4, 'POINT EMPTY'), \
                 (5, CAST(NULL AS TEXT))) \
                 SELECT ST_AsText(ST_BuildArea(ST_GeomFromText(wkt))) \
                 FROM t ORDER BY id",
            )
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let values = batch.column(0).as_string::<i32>();

        assert_eq!(values.value(0), "POLYGON((0 0,0 1,1 1,1 0,0 0))");
        assert!(values.is_null(1), "open linework should return NULL");
        assert!(values.is_null(2), "point-only input should return NULL");
        assert_eq!(values.value(3), "POLYGON EMPTY");
        assert!(values.is_null(4), "null input should propagate to NULL");
    }
}

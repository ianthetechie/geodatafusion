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
use geos::Geom;

use crate::data_types::any_single_geometry_type_input;
use crate::error::GeoDataFusionResult;

/// Returns a topological union of the input geometry components.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct UnaryUnion {
    coord_type: CoordType,
}

impl UnaryUnion {
    pub fn new(coord_type: CoordType) -> Self {
        Self { coord_type }
    }
}

impl Default for UnaryUnion {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

impl ScalarUDFImpl for UnaryUnion {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_unaryunion"
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
        Ok(unary_union_impl(args)?)
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Returns a topological union of the components of a geometry. For linework, intersections are noded. Overlapping segments (even e.g. in a MultiPolygon) are dissolved, provided that the individual components are valid. This is the unary version of ST_Union.",
                "ST_UnaryUnion(geometry)",
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

fn unary_union_impl(args: ScalarFunctionArgs) -> GeoDataFusionResult<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let geometry_array = from_arrow_array(&arrays[0], &args.arg_fields[0])?;

    let unions = geometry_array
        .as_ref()
        .to_geos()?
        .into_iter()
        .map(|maybe_geometry| match maybe_geometry {
            None => Ok(None),
            Some(geometry) => geometry.unary_union().map(Some),
        })
        .collect::<std::result::Result<Vec<_>, geos::Error>>()?;

    let output_type = GeoArrowType::from_arrow_field(args.return_field.as_ref())?;
    let GeoArrowType::Geometry(geometry_type) = output_type else {
        return Err(DataFusionError::Internal(
            "ST_UnaryUnion expected a Geometry return type".to_string(),
        )
        .into());
    };
    let result = GeometryArray::from_geos(unions, geometry_type)?;

    Ok(ColumnarValue::Array(result.to_array_ref()))
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::udf::geos::processing::BuildArea;
    use crate::udf::native::io::{AsText, GeomFromText};

    #[tokio::test]
    async fn nodes_crossing_linework_before_build_area() {
        let context = SessionContext::new();
        context.register_udf(UnaryUnion::default().into());
        context.register_udf(BuildArea::default().into());
        context.register_udf(GeomFromText::default().into());
        context.register_udf(AsText.into());

        let frame = context
            .sql(
                "SELECT ST_AsText(ST_BuildArea(ST_UnaryUnion(\
                    ST_GeomFromText('LINESTRING(0 0,2 2,0 2,2 0,0 0)'))))",
            )
            .await
            .expect("plan build area from noded crossing linework");
        let batch = frame
            .collect()
            .await
            .expect("build area from noded crossing linework")
            .into_iter()
            .next()
            .expect("one result row");
        let area = batch.column(0).as_string::<i32>();

        assert!(
            area.value(0).starts_with("MULTIPOLYGON("),
            "expected the noded bow-tie ring to become two polygons, got {}",
            area.value(0)
        );
    }
}

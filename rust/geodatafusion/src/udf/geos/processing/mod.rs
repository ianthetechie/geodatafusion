mod build_area;
mod line_merge;
mod unary_union;

#[cfg(feature = "geos-3_11")]
pub use build_area::BuildArea;
#[cfg(feature = "geos-3_11")]
pub use line_merge::LineMerge;
#[cfg(feature = "geos-3_11")]
pub use unary_union::UnaryUnion;

pub fn register(session_context: &datafusion::prelude::SessionContext) {
    #[cfg(feature = "geos-3_11")]
    session_context.register_udf(BuildArea::default().into());
    #[cfg(feature = "geos-3_11")]
    session_context.register_udf(LineMerge::default().into());
    #[cfg(feature = "geos-3_11")]
    session_context.register_udf(UnaryUnion::default().into());
}

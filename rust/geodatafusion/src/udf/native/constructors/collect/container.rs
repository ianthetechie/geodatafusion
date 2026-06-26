//! Native `GeometryTrait` wrapper used to emit collected groups with `GeometryBuilder`.
//!
//! The wrapper borrows already-classified group members and presents them as the chosen
//! MULTI*/GEOMETRYCOLLECTION container without flattening nested collections.

use datafusion::error::DataFusionError;
use geo_traits::{
    Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, UnimplementedLine, UnimplementedLineString,
    UnimplementedPoint, UnimplementedPolygon, UnimplementedRect, UnimplementedTriangle,
};
use geoarrow_array::scalar::{Geometry, LineString, Point, Polygon};

use crate::error::{GeoDataFusionError, GeoDataFusionResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CollectedKind {
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
}

/// Borrowed view over one collected output geometry.
pub(super) struct CollectedGeometry<'a> {
    members: &'a [Geometry<'a>],
    kind: CollectedKind,
    dim: Dimensions,
}

impl<'a> CollectedGeometry<'a> {
    pub(super) fn new(members: &'a [Geometry<'a>], kind: CollectedKind, dim: Dimensions) -> Self {
        debug_assert!(!members.is_empty());
        Self { members, kind, dim }
    }
}

impl GeometryTrait for CollectedGeometry<'_> {
    type T = f64;
    type PointType<'b>
        = UnimplementedPoint<f64>
    where
        Self: 'b;
    type LineStringType<'b>
        = UnimplementedLineString<f64>
    where
        Self: 'b;
    type PolygonType<'b>
        = UnimplementedPolygon<f64>
    where
        Self: 'b;
    type MultiPointType<'b>
        = CollectedGeometry<'b>
    where
        Self: 'b;
    type MultiLineStringType<'b>
        = CollectedGeometry<'b>
    where
        Self: 'b;
    type MultiPolygonType<'b>
        = CollectedGeometry<'b>
    where
        Self: 'b;
    type GeometryCollectionType<'b>
        = CollectedGeometry<'b>
    where
        Self: 'b;
    type RectType<'b>
        = UnimplementedRect<f64>
    where
        Self: 'b;
    type TriangleType<'b>
        = UnimplementedTriangle<f64>
    where
        Self: 'b;
    type LineType<'b>
        = UnimplementedLine<f64>
    where
        Self: 'b;

    fn dim(&self) -> Dimensions {
        self.dim
    }

    fn as_type(
        &self,
    ) -> GeometryType<
        '_,
        Self::PointType<'_>,
        Self::LineStringType<'_>,
        Self::PolygonType<'_>,
        Self::MultiPointType<'_>,
        Self::MultiLineStringType<'_>,
        Self::MultiPolygonType<'_>,
        Self::GeometryCollectionType<'_>,
        Self::RectType<'_>,
        Self::TriangleType<'_>,
        Self::LineType<'_>,
    > {
        match self.kind {
            CollectedKind::MultiPoint => GeometryType::MultiPoint(self),
            CollectedKind::MultiLineString => GeometryType::MultiLineString(self),
            CollectedKind::MultiPolygon => GeometryType::MultiPolygon(self),
            CollectedKind::GeometryCollection => GeometryType::GeometryCollection(self),
        }
    }
}

impl GeometryCollectionTrait for CollectedGeometry<'_> {
    type GeometryType<'b>
        = &'b Geometry<'b>
    where
        Self: 'b;

    fn num_geometries(&self) -> usize {
        self.members.len()
    }

    unsafe fn geometry_unchecked(&self, i: usize) -> Self::GeometryType<'_> {
        unsafe { self.members.get_unchecked(i) }
    }
}

impl MultiPointTrait for CollectedGeometry<'_> {
    type InnerPointType<'b>
        = Point<'b>
    where
        Self: 'b;

    fn num_points(&self) -> usize {
        self.members.len()
    }

    unsafe fn point_unchecked(&self, i: usize) -> Self::InnerPointType<'_> {
        match unsafe { self.members.get_unchecked(i) }.as_type() {
            GeometryType::Point(p) => p.clone(),
            // Valid because `classify_members` selected the multi-point container.
            _ => unreachable!("ST_Collect MultiPoint member was not a Point"),
        }
    }
}

impl MultiLineStringTrait for CollectedGeometry<'_> {
    type InnerLineStringType<'b>
        = LineString<'b>
    where
        Self: 'b;

    fn num_line_strings(&self) -> usize {
        self.members.len()
    }

    unsafe fn line_string_unchecked(&self, i: usize) -> Self::InnerLineStringType<'_> {
        match unsafe { self.members.get_unchecked(i) }.as_type() {
            GeometryType::LineString(ls) => ls.clone(),
            // Valid because `classify_members` selected the multi-line container.
            _ => unreachable!("ST_Collect MultiLineString member was not a LineString"),
        }
    }
}

impl MultiPolygonTrait for CollectedGeometry<'_> {
    type InnerPolygonType<'b>
        = Polygon<'b>
    where
        Self: 'b;

    fn num_polygons(&self) -> usize {
        self.members.len()
    }

    unsafe fn polygon_unchecked(&self, i: usize) -> Self::InnerPolygonType<'_> {
        match unsafe { self.members.get_unchecked(i) }.as_type() {
            GeometryType::Polygon(p) => p.clone(),
            // Valid because `classify_members` selected the multi-polygon container.
            _ => unreachable!("ST_Collect MultiPolygon member was not a Polygon"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AtomicBase {
    Point,
    LineString,
    Polygon,
}

fn atomic_base<M: GeometryTrait<T = f64>>(member: &M) -> Option<AtomicBase> {
    match member.as_type() {
        GeometryType::Point(_) => Some(AtomicBase::Point),
        GeometryType::LineString(_) => Some(AtomicBase::LineString),
        GeometryType::Polygon(_) => Some(AtomicBase::Polygon),
        _ => None,
    }
}

fn is_same_dimension(a: Dimensions, b: Dimensions) -> bool {
    use geoarrow_schema::Dimension;

    match (Dimension::try_from(a), Dimension::try_from(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Choose the output container kind and validate that all members share one coordinate dimension.
pub(super) fn classify_members<M: GeometryTrait<T = f64>>(
    members: &[M],
) -> GeoDataFusionResult<(CollectedKind, Dimensions)> {
    debug_assert!(!members.is_empty());

    let dim0 = members[0].dim();
    let base0 = atomic_base(&members[0]);
    let mut homogeneous_atomic = base0.is_some();

    for member in members {
        if !is_same_dimension(dim0, member.dim()) {
            return Err(GeoDataFusionError::DataFusion(DataFusionError::NotImplemented(
                "ST_Collect cannot combine geometries of differing coordinate dimensions (e.g. XY \
                 and XYZ)"
                    .to_string(),
            )));
        }
        if homogeneous_atomic && atomic_base(member) != base0 {
            homogeneous_atomic = false;
        }
    }

    let kind = match base0 {
        Some(AtomicBase::Point) if homogeneous_atomic => CollectedKind::MultiPoint,
        Some(AtomicBase::LineString) if homogeneous_atomic => CollectedKind::MultiLineString,
        Some(AtomicBase::Polygon) if homogeneous_atomic => CollectedKind::MultiPolygon,
        _ => CollectedKind::GeometryCollection,
    };
    Ok((kind, dim0))
}

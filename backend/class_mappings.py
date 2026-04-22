# MLRSNet 60 labels as identified from dataset research

MLRS_LABELS = [
    "airplane", "airport", "bare soil", "baseball diamond", "basketball court",
    "beach", "bridge", "buildings", "cars", "chaparral",
    "cloud", "containers", "crosswalk", "dense residential area", "desert",
    "dock", "factory", "field", "football field", "forest",
    "freeway", "golf course", "grass", "greenhouse", "gully",
    "habor", "intersection", "island", "lake", "mobile home",
    "mountain", "overpass", "park", "parking lot", "parkway",
    "pavement", "railway", "railway station", "river", "road",
    "roundabout", "runway", "sand", "sea", "ships",
    "snow", "snowberg", "sparse residential area", "stadium", "swimming pool",
    "tanks", "tennis court", "terrace", "track", "trail",
    "transmission tower", "trees", "water", "wetland", "wind turbine"
]

def get_class_label(index: int) -> str:
    """Return the label for a given class index."""
    if 0 <= index < len(MLRS_LABELS):
        return MLRS_LABELS[index]
    return f"Unknown ({index})"

def get_all_labels():
    """Return all labels."""
    return MLRS_LABELS

from cleaner.feature_extractor import extract_features_from_images

if __name__ == "__main__":
    features_dict = extract_features_from_images("images/circle/", "features/circle/", 64)

#!/usr/bin/env python3
"""


Copyright (c) 2025 Sheel Morjaria
License: CC BY-ND 4.0 International
Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Last Updated: December 27, 2025
"""

"""Add supervised classification capabilities to the framework."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data.io import FeatureIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_call_type_classifier():
    """Demonstrate how to add call type classification to the framework."""

    print("🎯 ADDING CALL TYPE CLASSIFICATION")
    print("=" * 50)

    # Load existing features
    feature_dir = Path("output/comprehensive_analysis/features")
    if not feature_dir.exists():
        print("❌ No features found. Run simple_analysis.py first.")
        return False

    print("📥 Loading extracted features...")
    features = FeatureIO.load_features(feature_dir)

    # Create synthetic call type labels for demonstration
    # In real use, these would come from human annotations or metadata
    print("\n🏷️  Creating Call Type Labels")
    print("-" * 30)

    call_types = []
    feature_vectors = []

    for i, feature_dict in enumerate(features):
        # Extract relevant features for classification
        feature_vector = []

        # Pitch-based features
        if "pitch" in feature_dict:
            pitch = feature_dict["pitch"]
            valid_pitch = pitch[~np.isnan(pitch)]
            if len(valid_pitch) > 0:
                feature_vector.extend(
                    [
                        np.mean(valid_pitch),  # Mean pitch
                        np.std(valid_pitch),  # Pitch variability
                        np.max(valid_pitch) - np.min(valid_pitch),  # Pitch range
                        len(valid_pitch) / len(pitch),  # Pitch coverage
                    ]
                )
            else:
                feature_vector.extend([0, 0, 0, 0])

        # Spectral features
        if "spectral" in feature_dict:
            spectral = feature_dict["spectral"]
            feature_vector.extend(
                [
                    spectral.get("spectral_centroid_mean", 0),
                    spectral.get("spectral_bandwidth_mean", 0),
                    spectral.get("spectral_rolloff_mean", 0),
                    spectral.get("zero_crossing_rate_mean", 0),
                ]
            )
        else:
            feature_vector.extend([0, 0, 0, 0])

        # Temporal features
        feature_vector.append(feature_dict.get("duration", 0))

        # Add MFCC statistics
        if "mfcc" in feature_dict:
            mfcc = feature_dict["mfcc"]
            # Use first 12 MFCC coefficients' statistics
            for coeff_idx in range(min(12, mfcc.shape[0])):
                feature_vector.extend([np.mean(mfcc[coeff_idx]), np.std(mfcc[coeff_idx])])
        else:
            feature_vector.extend([0] * 24)  # 12 coefficients * 2 stats

        feature_vectors.append(feature_vector)

        # Synthetic call type assignment based on features
        # In real use, replace with actual annotations
        if len(feature_vector) > 4:  # Has pitch data
            mean_pitch = feature_vector[0]
            pitch_range = feature_vector[2]
            duration = feature_vector[8]

            if mean_pitch > 2000 and pitch_range > 300:
                call_types.append("alarm")
            elif duration > 10:
                call_types.append("territorial")
            else:
                call_types.append("social")
        else:
            call_types.append("unknown")

    # Convert to arrays
    X = np.array(feature_vectors)
    y = np.array(call_types)

    print(f"📊 Feature matrix shape: {X.shape}")
    print("🏷️  Call type distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for call_type, count in zip(unique, counts):
        print(f"   {call_type}: {count} samples")

    # For demonstration with limited data, create augmented dataset
    print("\n🔄 Creating Augmented Dataset for Demonstration")
    print("-" * 30)

    # Augment data by adding noise (for demo purposes)
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        # Original sample
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        # Add 4 augmented versions with small random variations
        for _ in range(4):
            noise = np.random.normal(0, 0.05, X[i].shape)
            X_augmented.append(X[i] + noise)
            y_augmented.append(y[i])

    X_aug = np.array(X_augmented)
    y_aug = np.array(y_augmented)

    print(f"📈 Augmented dataset: {X_aug.shape[0]} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_aug
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"📚 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Test set: {X_test.shape[0]} samples")

    # Train classifiers
    print("\n🤖 Training Classification Models")
    print("-" * 30)

    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n🔧 Training {name}...")
        clf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = clf.predict(X_test_scaled)

        # Results
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"classifier": clf, "accuracy": accuracy, "predictions": y_pred}

        print(f"✅ {name} Accuracy: {accuracy:.3f}")

    # Detailed evaluation for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_model = results[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name}")
    print("-" * 30)

    print("📊 Classification Report:")
    print(classification_report(y_test, best_model["predictions"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_model["predictions"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_aug),
        yticklabels=np.unique(y_aug),
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    output_dir = Path("output/comprehensive_analysis")
    plt.savefig(output_dir / "call_type_classification.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance (for Random Forest)
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["classifier"]
        feature_names = (
            [
                "mean_pitch",
                "pitch_std",
                "pitch_range",
                "pitch_coverage",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "zero_crossing",
                "duration",
            ]
            + [f"mfcc_{i}_mean" for i in range(12)]
            + [f"mfcc_{i}_std" for i in range(12)]
        )

        importances = rf_model.feature_importances_

        # Plot top 10 features
        top_indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(10), importances[top_indices])
        plt.yticks(range(10), [feature_names[i] for i in top_indices])
        plt.xlabel("Feature Importance")
        plt.title("Top 10 Features for Call Type Classification")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save classifier for future use
    import joblib

    joblib.dump(
        {
            "classifier": best_model["classifier"],
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": best_model_name,
        },
        output_dir / "call_type_classifier.pkl",
    )

    print(f"\n💾 Model saved: {output_dir / 'call_type_classifier.pkl'}")
    print(f"📊 Confusion matrix: {output_dir / 'call_type_classification.png'}")
    print(f"📈 Feature importance: {output_dir / 'feature_importance.png'}")

    return True


def demonstrate_prediction():
    """Show how to use the trained classifier for new data."""

    print("\n🔮 CALL TYPE PREDICTION DEMO")
    print("=" * 40)

    classifier_path = Path("output/comprehensive_analysis/call_type_classifier.pkl")
    if not classifier_path.exists():
        print("❌ No trained classifier found. Run create_call_type_classifier first.")
        return

    import joblib

    model_data = joblib.load(classifier_path)

    classifier = model_data["classifier"]
    scaler = model_data["scaler"]

    # Example prediction for new audio
    print("📝 Example: Predicting call type for new audio features")
    print("   (This would normally come from real audio analysis)")

    # Simulate new audio features
    new_features = np.array(
        [
            [
                1850,  # mean_pitch (Hz)
                120,  # pitch_std
                400,  # pitch_range
                0.8,  # pitch_coverage
                2500,  # spectral_centroid
                800,  # spectral_bandwidth
                3000,  # spectral_rolloff
                0.15,  # zero_crossing
                5.2,  # duration
            ]
            + [0] * 24
        ]
    )  # MFCC features (zeros for demo)

    # Scale and predict
    new_features_scaled = scaler.transform(new_features)
    prediction = classifier.predict(new_features_scaled)

    # Get prediction probabilities if available
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(new_features_scaled)
        classes = classifier.classes_

        print(f"🎯 Predicted call type: {prediction[0]}")
        print("📊 Prediction probabilities:")
        for class_name, prob in zip(classes, probabilities[0]):
            print(f"   {class_name}: {prob:.3f}")
    else:
        print(f"🎯 Predicted call type: {prediction[0]}")


if __name__ == "__main__":
    try:
        success = create_call_type_classifier()

        if success:
            demonstrate_prediction()

            print("\n" + "🎯" * 25)
            print("CLASSIFICATION CAPABILITIES ADDED!")
            print("🎯" * 25)

            print("\n✅ NEW CAPABILITIES:")
            print("  • Supervised call type classification")
            print("  • Feature importance analysis")
            print("  • Model persistence and reuse")
            print("  • Prediction confidence scores")

            print("\n🚀 INTEGRATION COMPLETE:")
            print("  • Unsupervised clustering ✅")
            print("  • Pattern recognition ✅")
            print("  • Supervised classification ✅")
            print("  • Call type categorization ✅")

    except Exception as e:
        print(f"❌ Classification setup failed: {e}")
        import traceback

        traceback.print_exc()

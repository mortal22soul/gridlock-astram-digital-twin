# Changelog

All notable changes to the ASTraM Mobility Digital Twin Command Center will be documented in this file.

## [Unreleased]

### Added
- Dynamic baseline congestion feature: Simulator now uses historical average corridor congestion counts specific to the selected day of the week, rather than static end-of-dataset counts.
- Dynamic OSM spatial defaults: The simulator automatically infers the most common OSM Highway Class, Lane Count, and median Distance to Road based on the selected corridor.

### Changed
- Refactored ML model categorical encoding: Swapped `LabelEncoder` for native XGBoost categorical support (`enable_categorical=True`). This solves a critical bug where tree splits were treating unrelated categories as ordinal values.
- Re-scaled precipitation simulator: The maximum slider value now dynamically represents an extreme hourly downpour for Bangalore (50 mm/hr).

### Fixed
- Fixed administrative bulk-closure outlier bug: Capped short-term events (e.g., debris, VIP movement) to 12 hours during model training to prevent the regression model from artificially inflating expected clearing times to 21+ hours.
- Fixed timezone mismatch bug in Open-Meteo weather API merge: Converted naive UTC weather timestamps to IST (`Asia/Kolkata`) prior to merging with incident data, ensuring the ML model correctly learns the impact of non-zero rainfall on traffic clearing duration.

//! Python bindings for SwiftLLM
//!
//! This module provides Python bindings via PyO3 for the SwiftLLM inference engine.

use pyo3::prelude::*;

/// SwiftLLM Python module
#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // TODO: Add Python bindings for:
    // - Engine
    // - Config types
    // - Sampling parameters
    // - Request/Response types

    Ok(())
}

//! The lifecycle model's generated curriculum: the hard-case turns
//! (`generate`) and the splits assembled from the real reviewed rows and the
//! curriculum an independent review agreed with (`assemble`).

mod assemble;
mod generate;

pub use assemble::{assemble_curriculum, CurriculumAssembly};
pub use generate::{generate_curriculum, CurriculumGeneration};

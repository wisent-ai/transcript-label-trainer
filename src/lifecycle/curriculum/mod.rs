//! The lifecycle model's generated curriculum and splits: the hard-case turns
//! (`generate`), the splits assembled from the real reviewed rows and the
//! curriculum an independent review agreed with (`assemble`), and a fresh
//! day-held-out split of reviewed rows (`split`).

mod assemble;
mod generate;
mod split;

pub use assemble::{assemble_curriculum, CurriculumAssembly};
pub use generate::{generate_curriculum, CurriculumGeneration};
pub use split::{split_by_day, DaySplit};

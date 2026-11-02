# Stage 1 — Data preparation inputs

The upstream workflow maps RNA-seq gene symbols through Entrez identifiers to Recon3D gene identifiers and aggregates expression through gene–protein–reaction rules for GIMME model extraction. The study resolved 1,675 reference gene identifiers through MyGene.info.

This directory documents that input stage. The original gene-mapping and GIMME-extraction scripts are not included in the repository. Supply the prepared model pickles, mapping table and manifests before running the downstream FBA and biomarker scripts.

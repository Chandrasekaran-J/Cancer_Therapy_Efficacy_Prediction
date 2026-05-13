# Stage 1 — Data Preparation

Builds the gene-mapping table linking RNA-seq gene symbols to Recon3D reactions via a two-step chain:

```
HGNC symbol → Entrez base ID → Recon3D gene ID (EntrezID_AT1)
```

This mapping is required for GIMME score aggregation and biomarker backtracking.

**Output:** `gene_map.pkl` containing the lookup table (~1,865 entries).

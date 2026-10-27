# Real IGES files

Put real `.igs` / `.iges` files here (FreeCAD export, McMaster-Carr CAD download, SolidWorks/Fusion export) and run:

    pytest tests/test_iges.py -s

Each file is parsed and its units, entity counts, bounding box, candidate holes and skipped entities are printed.

The `occ_*.igs` files already here are real OpenCASCADE IGES exports (the CAD kernel FreeCAD uses), made with gmsh:

| file | true size (mm) | holes |
|---|---|---|
| occ_plate_100x60x12_4holes.igs | 100 × 60 × 12 | 4 × Ø8.5 |
| occ_shaft_150_dia40.iges | 150 × 40 × 40 (Ø40 + Ø30 step) | outer Ø40, Ø30 |
| occ_flange_d120.igs | Ø120 × 20 | 4 × Ø11, bore Ø60 |
| occ_filleted_block_80x50x20.igs | 80 × 50 × 20 with R5 fillets | — |
| occ_knob_cone_sphere.igs | 40 × 40 × 42 | — |

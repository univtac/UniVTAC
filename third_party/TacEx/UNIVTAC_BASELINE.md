# UniVTAC TacEx baseline

UniVTAC's Isaac Sim 5.1 migration is based on these exact upstream revisions:

- TacEx branch: `isaac-5-0`
- TacEx commit: `f2051944e469a961241271fbf6fb60e272fc336b`
- libuipc commit: `1a7e93ef68765e4d3c15d5583f5c387e89af5183`
- libuipc/muda commit: `8f9e17d8e76a658df3b6ffeeffbbc9ac47ac54bf`
- libuipc/SymEigen commit: `c72a0082e44b3b8062727b25c33ce7450f9fa933`
- Isaac Sim: 5.1
- Isaac Lab: 2.3.0
- Python: 3.11
- Pix2Pix checkpoint SHA-256:
  `140075a9e818814fd717562c003bc81c4a1960a73720debe71442e663d4da6f9`

The vendored libuipc tree is expanded rather than stored as a Git submodule. Its
two nested dependencies are expanded at the revisions above as well.

UniVTAC intentionally diverges from upstream in a few places:

- GelSight camera prims are authored inside the sensor USD. Camera config only
  overrides optical attributes on the existing prim (`spawn=None`).
- GelSight optical rendering is selected once at startup. Taxim is the default;
  Pix2Pix is an optional backend using the upstream pretrained checkpoint.
- Task actors select `rigid` or `deformable` explicitly and expose
  `set_pose(pose, soft=False)`. A hard pose write restores the rest shape and
  clears motion; a soft pose write drives the appropriate UIPC constraint.
- UIPC state is copied to Fabric exactly once before RTX sensor rendering.

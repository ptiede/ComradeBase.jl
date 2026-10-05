```@meta
CurrentModule = ComradeBase
```

# ComradeBase

ComradeBase defines the interface shared by the [Comrade](https://github.com/ptiede/Comrade.jl)
ecosystem: abstract model types and their traits, the domains models are evaluated on,
and the `IntensityMap` array type that holds images and visibilities.

- [Domains and maps](domains.md): image grids, point domains with `Pt`, `Ti` and `Fr`
  dims, and executors.
- [Polarized maps](polarization.md): the `Stokes` dim, coherency maps with feed dims, and
  conversion between them.
- [Sharding with Reactant](sharding.md): placing maps and domains on several devices.
- [API](api.md): every documented name.

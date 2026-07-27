# Validation tests

This directory contains the migration safety net for `ProcessSqlData-TS`.

Run the current suite from the repository root:

```powershell
python test/run_tests.py
```

Safe migration utilities can be exercised without running the legacy script:

```powershell
python migration_tools.py show-layout
python migration_tools.py validate-config
python migration_tools.py compare --expected <legacy-output> --actual <new-output>
```

The first test layer covers:

- strict loading of the untracked `config.local.toml`;
- mandatory read-only database configuration;
- script-anchored legacy output paths;
- safe staging and backup paths;
- directory-tree and CSV compatibility comparisons;
- the frozen list of all 14 legacy output directories;
- offline transformations, plots, classification, and ARIMA order selection;
- synthetic `legacy` and `corrected` stage 2-14 end-to-end runs.

Real database snapshots and generated production data must not be committed here.
Only synthetic fixtures are allowed in Git.

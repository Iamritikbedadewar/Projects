# Confidentiality and Offline-Execution Rules

## Non-negotiable rules

1. Keep the Matplus directory in a normal local folder that is not synchronized by
   OneDrive, Google Drive, Dropbox, Box, iCloud, Git, or another backup service.
2. Never copy Matplus JSON files into this project.
3. Never use the files in ChatGPT, an API, a cloud notebook, an online IDE, or a
   hosted inference endpoint.
4. Never commit raw data, generated examples, prompts containing raw examples,
   checkpoints trained on the data, or private prediction files.
5. Do not screen-share raw examples or place them in presentation slides.
6. Check the signed agreement before sharing aggregate metrics or trained weights.

## Safe execution sequence

### Phase A: online, no private data is opened

- Create the virtual environment.
- Install Python packages.
- Run `scripts/download_public_models.py`.
- Confirm that `models/MODEL_MANIFEST.json` exists.

The downloader has no option for a dataset path. It downloads public weights only.

### Phase B: offline, private data may be opened

- Disconnect the computer from networks if practical.
- Use `scripts/run_offline.ps1`.
- Keep offline-related environment variables unchanged.
- Store all private outputs only on the local machine.

The runner blocks network sockets inside its own Python process. This is defense in
depth, not a replacement for disconnecting the computer or following the agreement.

## Data minimization

Core experiments read only:

- `synthetic_nlq_abuse.json`
- `synthetic_nlq_mongo_pairs.json`

The material schema and example material records are not needed for the primary
classification/generation experiment and therefore are not opened. Ambiguous queries
are optional and are reserved for a local qualitative stress test.

By default, saved predictions contain only opaque example IDs, labels, and aggregate
scores. The `-SavePrivateErrors` option deliberately saves raw text for local error
analysis and therefore produces confidential files.

## Local-path validation

The runner rejects paths containing common cloud-sync markers such as `OneDrive`,
`Google Drive`, `Dropbox`, `iCloud`, and `Box Sync`. This simple check cannot detect
every backup or synchronization tool. You remain responsible for checking the folder
properties and agreement.

## Before sharing code or a report

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\privacy_audit.py
```

Then manually confirm:

- no Matplus JSON file is present;
- no screenshot exposes values;
- no private prediction file is present;
- no trained checkpoint is present;
- no example was copied from the confidential data;
- derived aggregate results are permitted by the agreement.

If uncertain, share nothing until the professor or Matplus contact confirms it.


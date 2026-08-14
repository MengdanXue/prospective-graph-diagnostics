# Route A Analysis Environment

Task 7 simulations use an isolated local virtual environment created with
Python 3.12.13. Direct analysis dependencies are pinned in
`requirements-analysis.txt`, and the resolved environment is recorded in
`requirements-analysis.lock.txt`:

- NumPy 2.3.5
- Matplotlib 3.11.1

Create the environment on Windows with:

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock.txt
```

All stochastic experiment and bootstrap seeds are frozen in
`configs/discriminability_grid.json`. Each result record also stores the Python
and package versions used for that run. The environment directory is ignored;
the pin file and per-record provenance are committed.

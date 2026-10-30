# data/

* `raw/`  — place `electricity.csv` (Elec2, River/OpenML schema: `date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, class`) here if the automatic download is not possible. **Not shipped with this project.**
* `processed/` — reserved for derived data; the study does not need any (preprocessing is done in memory).

Synthetic streams are generated on the fly (`src/data_streams.py`) from seeds in `configs/default.yaml`; nothing is stored.

How to obtain Elec2 on a machine with internet access:
```bash
python -c "from river import datasets; datasets.Elec2().download()"   # stores <river_data>/Elec2/electricity.csv
cp <river_data>/Elec2/electricity.csv data/raw/electricity.csv
```
The loader never substitutes another dataset; if the file is missing and the download fails, it raises `DatasetUnavailableError` with these instructions.

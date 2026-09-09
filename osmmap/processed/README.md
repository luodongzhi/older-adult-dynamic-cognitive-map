# OSM 标准地图包

来源：`../map.osm`，原始文件未修改。地图数据版权属于 OpenStreetMap contributors，许可为 ODbL 1.0。

## 可直接使用的文件

- `imported_map.json`：当前项目 `load_graph_json` 可以直接读取的完整地图；
- `roads.shp`：步行路网和入口连接线；
- `nodes.shp`：路网节点与入口节点；
- `aois.shp`：HOME、WORK、CAFE、PARK、SHOP 五类设施；
- `entrances.shp`：OSM明确入口与算法推断入口；
- `buildings.shp`：视觉遮挡物；
- `entrance_connectors.shp`：入口到步行路网的连接质量检查线；
- `map_qc.html` / `map_qc.json`：处理质量报告；
- `aoi_catalog.csv` / `entrance_catalog.csv`：可人工检查的对象目录；
- `agent_bindings.json`：三名老年Agent的住宅、日常锚点和共同线下会面点；
- `interventions.json`：算法为新地图生成的候选干预，不会覆盖正式实验干预；
- `pycharm_run_osm.example.json`：旧版单文件格式的参考示例，不会覆盖当前实验包。

正式运行读取 `experiments/current/`。地图处理完成后，请检查候选对象，再把确认后的内容复制到
`experiments/current/interventions.json`；不要把自动生成的候选直接当作已审定实验设计。

## 本次结果

- 节点：3246
- 道路：3459
- AOI：336
- 入口：341
- 建筑遮挡物：266
- OSM明确入口或门被设施采用：15
- 算法推断入口必须通过 `SOURCE` 与 `CONF` 字段识别，不能当作OSM真实入口。

如需重新处理：

```powershell
python tools/process_osm_map.py --input osmmap/map.osm --output osmmap/processed
```

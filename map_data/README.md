# 当前真实地图数据

- `road_real.shp`: 道路 Polyline，29 条记录，ID 字段为 `ROAD_ID`。
- `actual_map.shp`: AOI Polygon，12 条记录，字段为 `aoi_id`, `function`, `name`。
- `interventions.json`: t0 空间干预列表；目前为空，所以运行的是初始真实地图基线。
- `imported_map.json`: PyCharm 入口自动生成的地图缓存，不需要手工编辑。

修改 SHP 后再次运行 `run_pycharm.py`，缓存会根据文件修改时间自动重建。

当前三个 Agent 的住宅和工作 AOI 映射位于项目根目录 `pycharm_run.json`。

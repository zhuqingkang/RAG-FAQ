# 本地依赖环境：Milvus + Redis（docker compose）

本目录提供项目运行所需的两个外部服务：**Milvus（向量库）** 与 **Redis（缓存）**。
MySQL 与 DashScope 不在本文件范围内：前者需自行安装（`subjects_kg` 库 + `jpkb` 表），后者是云服务。

## 组件一览

| 服务 | 容器名 | 镜像 | 作用 | 端口 |
|---|---|---|---|---|
| `etcd` | milvus-etcd | etcd v3.5.5 | Milvus 的元数据存储 | 仅容器网络 |
| `minio` | milvus-minio | MinIO RELEASE.2023-03-20 | Milvus 的对象存储（段文件） | 仅容器网络 |
| `standalone` | milvus-standalone | milvus v2.4.4 | 向量数据库单机版 | **19530**（SDK）、9091（健康检查/指标） |
| `redis` | milvus-redis | redis | FAQ 问题列表与答案缓存 | **6379** |

> Milvus 单机版必须同时依赖 etcd 与 MinIO，三者是一个整体，不要单独停掉 etcd 或 MinIO。

## 快速开始

```bash
cd milvus_redis

# 启动（首次会拉取镜像，数分钟）
docker compose up -d

# 查看状态（等 4 个容器都是 running / healthy）
docker compose ps

# Milvus 健康检查，返回 OK 即就绪
curl http://localhost:9091/healthz

# 看某个服务日志（排查启动失败最有用）
docker compose logs -f standalone
```

停止与清理：

```bash
docker compose stop          # 停止，保留容器
docker compose down          # 删除容器与网络，./volumes 数据保留
docker compose down -v       # 同上，并删除匿名卷（绑定挂载 ./volumes 仍保留）
rm -rf volumes               # 彻底重置所有数据（谨慎）
```

目录结构（首次启动后生成）：

```
milvus_redis/
├── docker-compose.yml
├── README.md
└── volumes/          # 所有持久化数据，已被 .gitignore 忽略
    ├── etcd/
    ├── minio/
    ├── milvus/
    └── redis/
```

## 与项目配置的对应关系

`config.ini` 里的连接参数必须与 compose 暴露的端口、密码一致：

| `config.ini` 段 | 键 | 默认值 | 对应 compose |
|---|---|---|---|
| `[milvus]` | `host` / `port` | `localhost` / `19530` | `standalone.ports` 的 `19530:19530` |
| `[milvus]` | `database_name` | `test5` | Milvus 内部的数据库名，**需先创建** |
| `[milvus]` | `collection_name` | `edurag_final` | 集合名，不存在时由 `VectorStore` 自动建 |
| `[redis]` | `host` / `port` | `localhost` / `6379` | `redis.ports` 的 `6379:6379` |
| `[redis]` | `password` | `1234` | `redis.command` 里的 `--requirepass 1234` |

启动后建议先跑一遍连通性验证：

```python
# Milvus：列集合 + 按需建库
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
print(client.list_collections())
client.create_database("test5")   # 已存在会报错，可先 list_databases() 判断
```

```python
# Redis
import redis
r = redis.StrictRedis(host="localhost", port=6379, password="1234", decode_responses=True)
print(r.ping())                   # True
```

> 注意：`VectorStore` 初始化时会连 `database_name` 指定的库。若新建的 Milvus 里没有 `test5`，
> 会直接报库不存在——要么按上面代码建库，要么把 `config.ini` 改成已存在的库（如 `default`）。

## 可以修改哪些地方

| 想改什么 | 改哪里 | 说明与注意事项 |
|---|---|---|
| 镜像来源 / 版本 | 各服务的 `image:` | 文件里每个 `image` 下方都注释了官方镜像，拉不动或提示无权限时换成注释那行即可。升级 Milvus 大版本前先验证 `hybrid_search`、`FLOAT16_VECTOR` 字段与索引参数 |
| Milvus 对外端口 | `standalone.ports` 左值，如 `"19531:19530"` | 改完同步 `config.ini` 的 `[milvus] port` |
| Redis 对外端口 | `redis.ports` 左值 | 同上，同步 `[redis] port` |
| Redis 密码 | `redis.command` 的 `--requirepass 1234` | 改完必须同步 `config.ini` 的 `[redis] password`，否则报 `NOAUTH` |
| 数据落盘位置 | 各服务 `volumes` 左值 | 默认是相对 compose 文件的 `./volumes/*`；也可换绝对路径（Windows 写成 `D:/data/milvus`）或命名卷 |
| MinIO 账号密码 | `minio.environment` 的 `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 仅供 Milvus 内部使用，改完 `standalone` 会读同一组环境变量，无需额外配置 |
| 容器资源上限 | 各服务加 `mem_limit` / `cpus`，或用 `deploy.resources` | Milvus 单机建议 ≥ 4 GB 内存；内存不足会反复重启 |
| 网络名 | 文件末尾 `networks.default.name` | 只在需要与其它 compose 项目互通时才要改 |
| 时区 | 各服务 `environment` 加 `TZ=Asia/Shanghai` | 影响容器内日志时间戳，不影响业务 |
| 增加 MySQL | 追加一个 `mysql` 服务 | 本项目需要 `subjects_kg` 库与 `jpkb` 表，建表 SQL 见开发文档 §8.5 |

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `pull access denied` / `unauthorized` | 默认走阿里云个人版仓库，可能存在权限或网络差异 | 换成 `image` 下方注释里的官方镜像（`quay.io` / `minio` / `milvusdb` / `redis`） |
| 端口被占用（`port is already allocated`） | 本机已有 Milvus / Redis / 其它容器占用 19530、9091、6379 | 改端口映射左值，并同步 `config.ini` |
| `milvus-standalone` 反复重启 | etcd 或 MinIO 未就绪、内存不足 | `docker compose logs -f etcd minio standalone`；给 Docker 分配更多内存 |
| Milvus 报库不存在 | `config.ini` 的 `database_name=test5` 尚未创建 | 用上文 `create_database("test5")` 建库，或改用 `default` |
| Redis 连接报 `NOAUTH Authentication required` | 客户端没带密码，或密码与 compose 不一致 | 核对 `[redis] password` 与 `--requirepass` |
| 想彻底重来 | 旧数据干扰 | `docker compose down` → `rm -rf volumes` → `docker compose up -d` |
| 启动时提示 `version is obsolete` | Compose V2 已废弃 `version` 字段 | 仅告警，可删掉文件首行 `version: '3.5'` |

## 版本说明

本项目实测组合：**Milvus v2.4.4 + pymilvus 3.0.2 + redis 8.1 客户端**。
Milvus 与 pymilvus 的大版本绑定较紧，升级任一侧前建议先在小数据量下跑通「建集合 → 写入 → 混合检索」再切换。

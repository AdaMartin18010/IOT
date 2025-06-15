# 相关说明

## 1目录说明

### navigate/

所有的名字空间

### api/

与外界系统交互的接口定义实现

### common/

本包公用的定义实现

### config/

配置文件定义实现

### errors/

*统一的错误封装* 暂定

### global/

全局定义和操作,后续会逐步移除

### initialize/

 初始化相关

### internal/

内部应用和重构不稳定的包

### iot/

iot模型的封装和实现

### store/

数据库相关的定义实现

### pubsub/

订阅和发布模型的封装

### cmd/servers/

具体的命令行工具 各种server的实现

## 2 增加nats 消息中间件

### 2.1 topic sub/pub definition

#### 定义基础的服务类型和topic

***topic_charset*** **=def=** {  
*topic_char_set*  = [0-9a-zA-Z]  
*segmentation_char_set* = [.]  
*pattern_char_set* = [*>]  
}  

***definition***  
**publish_topic** ==== *topic_char_set* ***composite of*** *segmentation_char_set*  
**subscribe_topic** === *publish_topic* ***composite of*** ***only one of*** *pattern_char_set*  
***订阅只能使用pattern_char_set中的一个***  
***不然会带来各种不确定的路由分发(取决于mq的各种实现)***  
***从而会导致运维排错和软件系统运行时紊乱的问题***  

### 2.2 topic hierarchy definition

#### 定义topic 层次结构

rt.gzb.navl.1.in.sys.cpt.>  
rt.gzb.navl.1.in.sys.cpt.svr.playscreen.cmd  
rt.gzb.navl.1.in.sys.cpt.svr.navlstatus.status  
rt.gzb.navl.1.in.sys.cpt.dvr.radar.down.right.cmd  
rt.sx.navl.1.in.sys.cpt.dvr.radar.down.right.cmd  
**rt** -- root 占位 便以迁移和重定向  
[rt].>  
*gzb* -- 葛洲坝  
*sx* -- 山峡  
[rt].[gzb|sx].>  

**navl** -- navigation lock 船闸  
***1*** -- 船闸标识  
***in*** -- internal 内部  
***ex*** -- external 外部  
[rt].[gzb|sx].[1|2|3].[in|ex].>

*sys* -- system 系统标识  
如果未定义则使用sys占位 便于后续区分
船闸测速系统 定义为 ***navlsm*** short for  navigation lock shipspeed measure system.
[rt].[gzb|sx].[1|2|3].[in|ex].[sys|navlssm|user_defined].>  

*cpt* --  component 组件
*svc* --  svc service 抽象的软件服务
[rt].[gzb|sx].[1|2|3].[in|ex].[navlssm].[cpt|svc].>  

*svr* -- server 特定的服务器
*dvr* -- driver 特定的软件驱动
[rt].[gzb|sx].[1|2|3].[in|ex].[navlssm].[cpt].[svr|dvr].>  

*playscreen* -- 显示屏服务程序  
*platform* -- 平台服务程序  
*navlstatus* -- 通航船闸状态监测-服务程序  
*stopline* --  船闸禁停线监测-服务程序  
*shipspeed* -- 测量船舶速度-服务程序  
*waterline* -- 测量船舶吃水深-服务程序  
[rt].[gzb|sx].[1|2|3].[in].[navlssm].[cpt].[svr].[navlstatus|stopline|shipspeed|playscreen|waterline].>  

*radar* -- 雷达-驱动程序  
*relay* -- 开关量or继电器-驱动程序  
*tripodhead* -- 云台-驱动程序
*speedunit* -- 逻辑组合的测量单元-驱动程序 (例如 开关量,雷达和云台组合成一个速度测量的逻辑单位)  
[rt].[gzb|sx].[1|2|3].[in].[navlssm].[cpt].[dvr].[radar|relay|tripodhead|speedunit].>  

*up-down-left-right-top-bottom*  可以定义的具体物理现实位置  
*cmd-conf-status-event-action* 具体软件组件的静态和动态行为接口,区分订阅请求查询和存储分发不同的软件接口交互类型  
[rt].[gzb|sx].[1|2|3].[in].[navlssm].[cpt].[dvr].[radar].[up|down].[left|right].[cmd|conf|status|event|action]  
 [config]   **key/value properties for watch or setting.**  *待定义*  
 [action]   **publish action message**  *待定义*  
 [event]    **publish event message**  **defined事件**  
 [cmd]      **reply command for request**  **driver实现**  
 [status]   **reply status for requre**  **请求和回应**
 [state]    **publish current state for subcribe**  **订阅通知**
example:
按照抽象粒度和软件架构工程实现的角度划分如下:  

 ***rt.gzb.[1|2|3].in.sys.cpt.svr.navlstatus.[status|event]***  
status: 实现状态查询和回应
state: 状态变更订阅通知
event: 实现事件的订阅通知  
**具体的语义见程序实现的pubsub topic 说明**

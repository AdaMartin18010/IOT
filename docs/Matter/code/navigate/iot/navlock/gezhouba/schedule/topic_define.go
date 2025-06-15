package schedule

//统一定义基础的组件类型名称和名字
//消息队列
const (

	//统一消息主题后集中处理
	MsgTopic_NavlockStatus = "sys.cpt.svr.navlstatus.state.navl."
	MsgTopic_NavStatus     = "sys.cpt.svr.navlstatus.state.nav"

	// playscreen 播放屏需要的 速度数据结构
	MsgTopic_PlayScreen_ShipSpeeds = "sys.cpt.svr.shipspeed.event.shipspeeds"

	////platform
	//MsgTopic_Platform_Stopline   = "Navlock/gezhouba/Msg/Platform/Stopline"
	//MsgTopic_Platform_Speedlimit = "Navlock/gezhouba/Msg/Platform/Speedlimit"

	// //web
	// MsgTopic_OverSpeed = "sys.cpt.svr.shipspeed"
	// MsgTopic_StopLine  = "sys.cpt.svr.stopline"
)

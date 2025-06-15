package playscreen

import (
	"bytes"
	"context"
	"net/http"

	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
)

// 清空盒子里的节目
// http://192.168.42.129/api/clrprgms
// Method: DELETE
func (ps *PlayScreen) doClearLanProgram(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/clrprgms"
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		ps.httpClearLanProgram.Logger.Sugar().Errorf("doClearLanProgram err:%+v.%s", err, ps.Info())
	}
	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpClearLanProgram.Logger.Sugar().Debugf("url: %s.%s",
			url,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpClearLanProgram.Logger.Sugar().Errorf("doClearLanProgram, err:%+v.%s", err, ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

// http://192.168.42.129/api/program/program_name.vsn
// Method: POST
// Content-type: application/json;charset=utf-8
func (ps *PlayScreen) doPublishProgramDefault(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/program/" + ps.programDefaultName + ".vsn"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(ps.programDefaultJsonContent))
	if err != nil {
		ps.httpPublishProgramDefault.Logger.Sugar().Errorf("doPublishProgramDefault err:%+v.%s", err, ps.Info())
	}
	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpPublishProgramDefault.Logger.Sugar().Debugf("url: %s ,programDefaultName:%s , programDefaultJsonContent: %s.%s",
			url,
			ps.programDefaultName,
			ps.programDefaultJsonContent,
			ps.Info())
	}
	resp, err := http.DefaultClient.Do(req)

	if err != nil {
		ps.httpPublishProgramDefault.Logger.Sugar().Errorf("programDefaultJsonContent:%s, err:%+v.%s",
			ps.programDefaultJsonContent,
			err,
			ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

// http://192.168.42.129/api/program/program_name.vsn
// Method: POST
// Content-type: application/json;charset=utf-8
func (ps *PlayScreen) doPublishProgramSpeed(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/program/" + ps.programSpeedName + ".vsn"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(ps.programSpeedJsonContent))
	if err != nil {
		ps.httpPublishProgramSpeed.Logger.Sugar().Errorf("doPublishProgramSpeed err:%+v.%s", err, ps.Info())
	}
	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpPublishProgramSpeed.Logger.Sugar().Debugf("url: %s ,programSpeedName:%s , programSpeedJsonContent: %s.%s",
			url,
			ps.programSpeedName,
			ps.programSpeedJsonContent,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpPublishProgramSpeed.Logger.Sugar().Errorf("doPublishProgramSpeed --- programSpeedJsonContent:%s, err:%+v.%s",
			ps.programSpeedJsonContent,
			err, ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

// http://192.168.42.129/api/vsns/sources/lan/vsns/new.vsn/activated
// Method: PUT
// Content-type: application/json;charset=utf-8
// 描述：
//   - 该接口中lan表示节目的来源；
//   - 节目来源的类型如下：
//     lan: 该节目从本地网络发布，包括WiFi / LAN / USB线；
//     usb: 节目位于外部USB存储器中（使用了U盘扩容）；
//     usb-synced: 节目是从外部USB存储设备同步（复制）到内部存储设备的；
//     internet: 该节目来自Internet；
//   - 该接口中"new.vsn"是要切换到的节目的名称；
//   - 如果是通过usb线发布的节目，那么该节目的来源是"lan"；
//
// response:
// {\n \"playing\": {\n  \"type\": \"lan\",\n  \"name\": \"normal.vsn\"\n }\n}
func (ps *PlayScreen) doActiveProgramDefault(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/vsns/sources/lan/vsns/" + ps.programDefaultName + ".vsn/activated"
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, nil)
	if err != nil {
		ps.httpActiveProgramDefault.Logger.Sugar().Errorf("doActiveProgramDefault err:%+v.%s", err, ps.Info())
	}
	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpActiveProgramDefault.Logger.Sugar().Debugf("url: %s ,programDefaultName:%s.%s",
			url,
			ps.programDefaultName,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpActiveProgramDefault.Logger.Sugar().Errorf("doActiveProgramDefault err:%+v.%s", err, ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

// Response:
// {\n \"playing\": {\n  \"type\": \"lan\",\n  \"name\": \"speed.vsn\"\n }\n}
func (ps *PlayScreen) doActiveProgramSpeed(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/vsns/sources/lan/vsns/" + ps.programSpeedName + ".vsn/activated"
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, nil)
	if err != nil {
		ps.httpActiveProgramSpeed.Logger.Sugar().Errorf("doActiveProgramSpeed err:%+v.%s", err, ps.Info())
	}
	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpActiveProgramSpeed.Logger.Sugar().Debugf("url: %s ,programSpeedName:%s.%s",
			url,
			ps.programSpeedName,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpActiveProgramSpeed.Logger.Sugar().Errorf("doActiveProgramSpeed err:%+v.%s", err, ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

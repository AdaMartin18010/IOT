package playscreen

import (
	"bytes"
	"context"
	"net/http"
	"time"

	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
)

// PUT http://192.168.42.129/api/program/program_name.vsn
// Content-Type:application/json;charset=UTF-8
// 参数说明
//
//	program_name: 更新的节目名，必须是当前播放的节目名。
//	HTTP方法是PUT
//	Content-Type:application/json;charset=UTF-8
//	who是需要更新的 单行文本的坐标 : {"x", 0, "y", 0}
func (ps *PlayScreen) doUpdateProgramDefaultMultiLines(ctx context.Context) *drs_once.Result {
	url := ps.Cnf.Url + "/api/program/" + ps.programDefaultName + ".vsn"

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewBufferString(ps.programDefaultUpdateLinesJsonContent))
	if err != nil {
		ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Errorf("doUpdateProgramDefaultMultiLines err:%+v.%s",
			err,
			ps.Info())
	}

	req.Header.Set("Content-Type", "application/json;charset=utf-8")

	if ps.Cnf.EnableDebug {
		ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Debugf("url : %s ,programDefaultUpdateLinesJsonContent : %s.%s",
			url,
			ps.programDefaultUpdateLinesJsonContent,
			ps.Info())
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Errorf("programDefaultUpdateLinesJsonContent:%s, err:%+v.%s",
			ps.programDefaultUpdateLinesJsonContent,
			err,
			ps.Info())
	}

	return &drs_once.Result{
		Val: resp,
		Err: err,
	}
}

func (ps *PlayScreen) startUpdateProgramDefaultMultiLines() {
	if !ps.httpUpdateProgramDefaultMultiLines.IsRunning() {
		if ps.Cnf.EnableDebug {
			ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpUpdateProgramDefaultMultiLines.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpUpdateProgramDefaultMultiLines.Ctrl().Context())
		}

		ps.httpUpdateProgramDefaultMultiLines.Ctrl().WithTimeout(ps.Ctrl().Context(), time.Duration(ps.Cnf.ConnectReadTimeout)*time.Millisecond)

		if ps.Cnf.EnableDebug {
			ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Debugf(`ps.Ctrl().Context():%+v; httpUpdateProgramDefaultMultiLines.Ctrl().Context():%+v"`,
				ps.Ctrl().Context(), ps.httpUpdateProgramDefaultMultiLines.Ctrl().Context())
		}

		if err := ps.httpUpdateProgramDefaultMultiLines.Start(); err != nil {
			ps.httpUpdateProgramDefaultMultiLines.Logger.Sugar().Errorf(`Start() error: %+v.%s`, err, ps.httpUpdateProgramDefaultMultiLines.Info())
		}
	}
}

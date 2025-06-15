package playscreen

import (
	"context"
	"errors"
	"io"
	"net/http"

	drs_once "navigate/iot/navlock/gezhouba/model/driver/once"
)

// 处理未释放的资源 和 读取回应体 判断 回应状态
func (ps *PlayScreen) dealWithHttpResponseBodyAndStatusCode200(wk *drs_once.Once, result *drs_once.Result) bool {
	defer func() {
		//防止所有未释放资源产生
		if result != nil && result.Val != nil {
			resp, ok := result.Val.(*http.Response)
			if ok && resp != nil {
				resp.Body.Close()
			}
		}
		result = nil
	}()

	if result.Err != nil {
		if errors.Is(result.Err, context.Canceled) {
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			wk.Logger.Sugar().Errorf(`http - request time out,err : %+v.%s`, result.Err, ps.Info())
			return false
		}

		wk.Logger.Sugar().Errorf(`err : %+v.%s`, result.Err, ps.Info())
		return false
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			wk.Logger.Sugar().Errorf("tranform from interface failed.%s", ps.Info())
			return false
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			wk.Logger.Sugar().Errorf(`Error reading response body: %+v.%s`, err, ps.Info())
			return false
		}

		if resp.StatusCode == 200 {
			//成功的情况下 只有http 200
			if ps.Cnf.EnableDebug {
				wk.Logger.Sugar().Debugf(`Response StatusCode: %d,Responsebody: %s.%s`,
					resp.StatusCode, body, ps.Info())
			}
			return true
		} else {
			if ps.Cnf.EnableDebug {
				wk.Logger.Sugar().Debugf(`Response StatusCode: %d,Responsebody: %s.%s`,
					resp.StatusCode, body, ps.Info())
			}
		}
		body = nil
	}

	return false
}

// 处理未释放的资源 和 不读取回应体 判断 回应状态
func (ps *PlayScreen) dealWithHttpResponseStatusCode200(wk *drs_once.Once, result *drs_once.Result) bool {
	defer func() {
		//防止所有未释放资源产生
		if result != nil && result.Val != nil {
			resp, ok := result.Val.(*http.Response)
			if ok && resp != nil {
				resp.Body.Close()
			}
		}
		result = nil
	}()

	if result.Err != nil {
		if errors.Is(result.Err, context.Canceled) {
			return false
		}

		if errors.Is(result.Err, context.DeadlineExceeded) {
			wk.Logger.Sugar().Errorf(`http - request time out, err : %+v.%s`, result.Err, ps.Info())
			return false
		}

		wk.Logger.Sugar().Errorf(`err : %+v.%s`, result.Err, ps.Info())
		return false
	} else {
		resp, tranformOK := result.Val.(*http.Response)
		if !tranformOK {
			// program error
			wk.Logger.Sugar().Errorf("tranform from interface failed.%s", ps.Info())
			return false
		}
		if resp.StatusCode == 200 {
			//成功的情况下 只有http 200
			if wk.EnableDebug {
				wk.Logger.Sugar().Debugf(`Response StatusCode: %d.%s`,
					resp.StatusCode, wk.Info())
			}
			return true
		} else {
			if wk.EnableDebug {
				wk.Logger.Sugar().Debugf(`Response StatusCode: %d.%s`,
					resp.StatusCode, wk.Info())
			}
		}
	}

	return false
}

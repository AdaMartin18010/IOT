package platform_test

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"testing"
	"time"

	cm "navigate/common/model"
	bs "navigate/internal/bytestr"
	api_pf "navigate/iot/navlock/gezhouba/api/platform"

	openssl "github.com/forgoer/openssl"
)

func EnDecryptWraper(src []byte) {
	fmt.Println("-----------------------------------")
	key := []byte("lB2BxrJdI4UUjK3KEZyQ0obuSgavB1SYJuAFq9oVw0Y=")
	iv := []byte("6lra6ceX26Fazwj1R4PCOg==")
	key, err := base64.StdEncoding.DecodeString(bs.BytesToString(key))
	if err != nil {
		fmt.Printf("base64.StdEncoding.DecodeString key err is : %+v\n", err)
	}
	iv, err = base64.StdEncoding.DecodeString(bs.BytesToString(iv))
	if err != nil {
		fmt.Printf("base64.StdEncoding.DecodeString  iv err is : %+v\n", err)
	}

	tgt, err := openssl.AesCBCEncrypt(src, key, iv, openssl.PKCS7_PADDING)
	if err != nil {
		fmt.Printf("err is : %+v\n", err)
	} else {
		fmt.Printf("src is : %s\n", bs.BytesToString(src))
		fmt.Printf("Encrypt src -> tgt is : OK\n")
		str := base64.StdEncoding.EncodeToString(tgt)
		fmt.Printf("base64 encode -> tgt is : %s\n", str)
	}

	tgt, err = openssl.AesCBCDecrypt(tgt, key, iv, openssl.PKCS7_PADDING)
	if err != nil {
		fmt.Printf("err is : %+v\n", err)
	} else {
		fmt.Printf("Decrypt -> tgt is : %s\n", bs.BytesToString(tgt))
	}
}

func DecryptWraper(src []byte) {
	fmt.Println("-----------------------------------")
	key := []byte("lB2BxrJdI4UUjK3KEZyQ0obuSgavB1SYJuAFq9oVw0Y=")
	iv := []byte("6lra6ceX26Fazwj1R4PCOg==")
	key, err := base64.StdEncoding.DecodeString(bs.BytesToString(key))
	if err != nil {
		fmt.Printf("base64.StdEncoding.DecodeString key err is : %+v\n", err)
	}
	iv, err = base64.StdEncoding.DecodeString(bs.BytesToString(iv))
	if err != nil {
		fmt.Printf("base64.StdEncoding.DecodeString  iv err is : %+v\n", err)
	}

	src, err = base64.StdEncoding.DecodeString(bs.BytesToString(src))
	if err != nil {
		fmt.Printf("base64.StdEncoding.DecodeString  src err is : %+v\n", err)
	} else {
		tgt, err := openssl.AesCBCDecrypt(src, key, iv, openssl.PKCS7_PADDING)
		if err != nil {
			fmt.Printf("AesCBCDecrypt err is : %+v\n", err)
		} else {
			fmt.Printf("Decrypt ->src - tgt is : %s\n", bs.BytesToString(tgt))
		}
	}

}

func TestDecrypt(t *testing.T) {

	src := []byte(`{"TableName":"overspeed","Where":"","Order":"","IsAsc":true,"IsCN":true,"Page":1,"Limt":10}`)
	EnDecryptWraper(src)
	src = []byte(`{"TableName":"overspeed","TableColumn":{"optime":"时间","lockname":"船闸名称","lockno":"闸次号","earlywarning":"预警","sailspeed":"航行速度"}}`)
	EnDecryptWraper(src)
	src = []byte(`{"TableName":"overspeed","DATA":[{"optime":"2020/7/31 14:00","lockname":"葛洲坝1#","lockno":"1","earlywarning":"超速","sailspeed":"2.00M/S"},
									{"optime":"2020/8/1 14:20","lockname":"葛洲坝2#","lockno":"4","earlywarning":"超速","sailspeed":"1.90M/S"},
									{"optime":"2020/8/1 04:00","lockname":"葛洲坝3#","lockno":"9","earlywarning":"超速","sailspeed":"1.95M/S"}]}`)
	EnDecryptWraper(src)
	src = []byte(`{"TableName":"overspeed","Where":"","Order":"","IsAsc":true,"IsCN":true,"Page":1,"Limt":10}`)
	EnDecryptWraper(src)
	src = []byte(`cCeAY+NVgRUEJrTu6QgCYnJ7ZhdR2uSyx5Je2dp3IIG4o8NlpIgSKoopH3D3cYnVOtcGKk2kBSPBDohSr5cdRZ1jNhrWf90KXo0GI2nJQwRl8p9bYrbcHtnDphMmtVxQuX0hjd/NmJrLVSNhOUg+Jm7UmKWbZhHLMs7Tfti7KMtJfXyuOsjGJwRnIBT7eMJL+XoN/PRcHLLnkxpW2yRzlpER3EqgsOhvFft2tquAZOypmRSTr8kc+TLzYVgCTZzSJZcsaxGtfJQSrOX7jIeiC+iTLMOMyuBTIq+pr53bGgVTOgQ+/d8hhFVITwenGprOBOslBhm7/2Dc5m0qppsjXtpTJ3sCnUgolpdpDEnjdglDLmoeyz2JPGoprxPoU/fhGJ2hCnjcl6AOSAqCA4JuVC9IvzNpAK2Qcc5v2LwTZHA25SuOh6M9d228FO4PlPCRTyQoCcOXYtAGQlAPKEJEFl/k5gR0gUbEYSZi0S05SIIuJk4TLzQvBZCijIv2AOJ2tAZM9en/QRaqCGWBflUxvp7FG97o9Pl2SEcUHoV8iLXg6BRI7DBEwLnziMTf+PazbTm3T/YNsDIKJTC0U3/+zV2wz4CWXZsv3Qg3js3/jH27arXvCNc08SwPf7OssSKWO5bKrM2S3jJiYuCRAdWE0+4u56aI8XXGz4xsd2V6yjmWPRoBo5gS373UTTCaFQ5oJbQG96a36Q/TZN0jN7pB2cMZSFWXezIK4K1QfMz/NC6UUuZ+7td86hsuj6cM4jtd2gtRDEv8kyi4DrGiW5FUZoiSgFYV4Sm3kqLlPvuAlQIfKa6kikA8vmBAxF3/384q`)
	DecryptWraper(src)

}

func EnDecrypt(t *testing.T, src []byte) {
	t.Logf("--------------------------------------------------------------------")
	dst, err := api_pf.EncryptEncodeWraper(src)
	if err != nil {
		t.Fatalf("EncryptEncodeWraper error: %+v", err)
	}
	t.Logf("EncryptEncodeWraper : %s", bs.BytesToString(dst))

	src1, err := api_pf.DecodeDecryptWraper(dst)
	if err != nil {
		t.Fatalf("DecodeDecryptWraper error: %+v", err)
	}
	t.Logf("DecodeDecryptWraper : %s", bs.BytesToString(src1))

	if bytes.Equal(src, src1) {
		t.Log(" src == dst")
	} else {
		t.Fatalf("src != dst mismatch")
	}
}

func TestEnDecrypts(t *testing.T) {
	src := []byte(`{"TableName":"overspeed","Where":"","Order":"","IsAsc":true,"IsCN":true,"Page":1,"Limt":10}`)
	EnDecrypt(t, src)

	src = []byte(`{"TableName":"overspeed","TableColumn":{"optime":"时间","lockname":"船闸名称","lockno":"闸次号","earlywarning":"预警","sailspeed":"航行速度"}}`)
	EnDecrypt(t, src)

	src = []byte(`{"TableName":"overspeed","DATA":[{"optime":"2020/7/31 14:00","lockname":"葛洲坝1#","lockno":"1","earlywarning":"超速","sailspeed":"2.00M/S"},{"optime":"2020/8/1 14:20","lockname":"葛洲坝2#","lockno":"4","earlywarning":"超速","sailspeed":"1.90M/S"},{"optime":"2020/8/1 04:00","lockname":"葛洲坝3#","lockno":"9","earlywarning":"超速","sailspeed":"1.95M/S"}]}`)
	EnDecrypt(t, src)

	src = []byte(`{"TableName":"overspeed","TableColumn":{"optime":"时间","lockname":"船闸名称","lockno":"闸次号","earlywarning":"预警","sailspeed":"航行速度"}}`)
	EnDecrypt(t, src)

	t.Log("--------------------DecodeDecryptWraper--------------")
	src = []byte(`cCeAY+NVgRUEJrTu6QgCYnJ7ZhdR2uSyx5Je2dp3IIG4o8NlpIgSKoopH3D3cYnVOtcGKk2kBSPBDohSr5cdRZ1jNhrWf90KXo0GI2nJQwRl8p9bYrbcHtnDphMmtVxQuX0hjd/NmJrLVSNhOUg+Jm7UmKWbZhHLMs7Tfti7KMtJfXyuOsjGJwRnIBT7eMJL+XoN/PRcHLLnkxpW2yRzlpER3EqgsOhvFft2tquAZOypmRSTr8kc+TLzYVgCTZzSJZcsaxGtfJQSrOX7jIeiC+iTLMOMyuBTIq+pr53bGgVTOgQ+/d8hhFVITwenGprOBOslBhm7/2Dc5m0qppsjXtpTJ3sCnUgolpdpDEnjdglDLmoeyz2JPGoprxPoU/fhGJ2hCnjcl6AOSAqCA4JuVC9IvzNpAK2Qcc5v2LwTZHA25SuOh6M9d228FO4PlPCRTyQoCcOXYtAGQlAPKEJEFl/k5gR0gUbEYSZi0S05SIIuJk4TLzQvBZCijIv2AOJ2tAZM9en/QRaqCGWBflUxvp7FG97o9Pl2SEcUHoV8iLXg6BRI7DBEwLnziMTf+PazbTm3T/YNsDIKJTC0U3/+zV2wz4CWXZsv3Qg3js3/jH27arXvCNc08SwPf7OssSKWO5bKrM2S3jJiYuCRAdWE0+4u56aI8XXGz4xsd2V6yjmWPRoBo5gS373UTTCaFQ5oJbQG96a36Q/TZN0jN7pB2cMZSFWXezIK4K1QfMz/NC6UUuZ+7td86hsuj6cM4jtd2gtRDEv8kyi4DrGiW5FUZoiSgFYV4Sm3kqLlPvuAlQIfKa6kikA8vmBAxF3/384q`)
	dst, err := api_pf.DecodeDecryptWraper(src)
	if err != nil {
		t.Fatalf("DecodeDecryptWraper error: %+v", err)
	}
	ss := []byte(`{"list":[{"时间":"2020/7/31 14:00","船闸名称":"葛洲坝1#","闸次号":"1","预警":"超速","航行速度":"2.00M/S","创建时间":"2022/7/31 16:01:02","编号":"29379724-6406-430f-ac9f-49a782ff2ba8"},{"时间":"2020/8/1 14:20","船闸名称":"葛洲坝2#","闸次号":"4","预警":"超速","航行速度":"1.90M/S","创建时间":"2022/7/31 16:01:02","编号":"f8a661c0-d840-4b65-b50b-9fd5d02f6966"},{"时间":"2020/8/1 04:00","船闸名称":"葛洲坝3#","闸次号":"9","预警":"超速","航行速度":"1.95M/S","创建时间":"2022/7/31 16:01:02","编号":"17826aa1-ed53-47a3-84ff-3d84a549dc0a"}],"count":3}`)
	if bytes.Equal(dst, ss) {
		t.Logf("src:%s,\nDecodeDecryptWraper(src):%s,\nsrc-str:%s,\nstr = dst",
			bs.BytesToString(src),
			bs.BytesToString(dst), bs.BytesToString(ss))
	} else {
		t.Fatalf("dst != ss")
	}
}

func UnmarshalResponse(t *testing.T, ss []byte, resp *api_pf.Response) {
	err := cm.Json.Unmarshal(ss, &resp)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	} else {
		t.Logf("response : %#v", resp)
	}
}

func TestJson(t *testing.T) {
	ss0 := []byte(`{"flag":true,"err":"","data":"调用成功!"}`)
	ss1 := []byte(`{"flag":true,"err":"","data":{"token":"hhhh1231231ppp+++afhaghu","expiration":"2022-08-14 01:00:00","type":"Bearer"}}`)
	var resp api_pf.Response
	UnmarshalResponse(t, ss0, &resp)
	if resp.Flag {
		ss, ok := resp.Data.(string)
		if ok {
			t.Logf("resp.Data : %s \n", ss)
		} else {
			t.Fatalf("resp.Data is not a string type")
		}
	}

	UnmarshalResponse(t, ss1, &resp)
	var tmsg api_pf.HttpTokenMsg
	t.Logf("TokenMsg is time to get token: %t", tmsg.IsTimeToGetToken())
	err := resp.ParseTokenMsg(&tmsg)
	if err != nil {
		t.Fatalf("resp.ParseTokenMsg error: %#v", err)
	}
	t.Logf("TokenMsg : %#v", tmsg)
	t.Logf("TokenMsg is time to get token: %t", tmsg.IsTimeToGetToken())
	t.Logf("time.Until(TokenMsg.Expiration) is: %f", time.Until(tmsg.Expiration).Minutes())
	t.Logf("TokenMsg.Expiration is: %s", tmsg.Expiration.String())

	// var tb api_pf.Table
	// ss, err := tb.SpeedlimitJson()
	// if err != nil {
	// 	t.Fatalf("SpeedlimitJson err: %#v", err)
	// }
	// t.Logf("Table json: %s", bs.BytesToString(ss))
	// t.Logf("Table SpeedlimitStr: %s", tb.SpeedlimitStr())
	// t.Logf("Table StoplineStr: %s", tb.StoplineStr())
}

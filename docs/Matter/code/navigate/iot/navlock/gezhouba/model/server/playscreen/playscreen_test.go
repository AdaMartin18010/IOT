package playscreen_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	cmpt "navigate/common/model/component"
	gzb_cf "navigate/config/iot/navlock/gezhouba"
	g "navigate/global"
	ibyte "navigate/internal/bytestr"
	gzb_ps "navigate/iot/navlock/gezhouba/model/server/playscreen"
	gzb_sche "navigate/iot/navlock/gezhouba/schedule"
	gzb_md "navigate/iot/navlock/model"

	jsoniter "github.com/json-iterator/go"

	"github.com/spf13/viper"
)

func TestComposite(t *testing.T) {
	ServerMaster := gzb_md.NewServerRoot(g.MainCtr)
	mcs01 := gzb_md.NewCptSt(gzb_md.NavlockModelKind, cmpt.IdName("navlock#1"), 1, ServerMaster.Ctrl().ForkCtxWg())
	mss01 := gzb_md.NewModelSysSt(mcs01.CmptInfo(), "#1", "ktest", "itest", "#1")
	NavlockModel_01 := gzb_md.NewNavlockModel(mcs01, mss01)

	ServerMaster.Cpts.AddCpts(NavlockModel_01)
	t.Logf("ServerMaster.Components len: %d", ServerMaster.Cpts.Len())
	mcs := NavlockModel_01.Fork(gzb_ps.Kind, cmpt.IdName("playscreen"))
	mss := gzb_md.NewModelSysSt(mcs.CmptInfo(), "navlock#1", "navlock", "#1", "playscreen")
	dr := gzb_md.NewBaseCptCtx(mcs, mss)
	trks := gzb_sche.NewTrackingSt(1, 5*time.Second, true)
	ps := gzb_ps.NewPlayScreenSvr(dr, trks)

	ps.Cnf = &gzb_cf.PlayScreenConfig{
		Enable:             true,
		EnableAutoTest:     true,
		EnableDebug:        true,
		Url:                "http://httpbin.org/post",
		ConnectReadTimeout: 3000,
		DoTimeInterval:     100,
	}

	NavlockModel_01.Cpts.AddCpts(ps)
	ServerMaster.Start()
	time.Sleep(5 * time.Second)
	ServerMaster.Stop()

	ps.Ctrl().WithCtx(NavlockModel_01.Ctrl().Context())
	ps.Start()
	ps.Stop()
	ps.Ctrl().WithCtx(NavlockModel_01.Ctrl().Context())
	ps.Start()
	ps.Stop()
	ServerMaster.Finalize()
}

// 不做配置文件判断
func TestNavl01Templete(t *testing.T) {
	navl01Viper := viper.New()
	navl01Viper.SetConfigType("json")
	var templete01 = []byte(`
{
	"Programs": {
		"Program": {
			"Information":{  
				"Width":  "288",
				"Height": "160"
			},
			"Pages": [
				{
					"Name":"main",
					"Regions": [
						{
							"Name": "main",
							"Layer": "1",
							"Rect": {
								"X": "0",
								"Y": "0",
								"Width": "288",
								"Height": "160"
							},
							"Items": [
								{
									"Name": "main",
									"Type": "4",
									"Text" : "注意安全",
									"TextColor": "#FFF0F5",
									"Speed": "15",
									"CenteralAlign": "1",
									"IsScroll": "1",
									"LogFont": {
										"lfHeight": "90",
										"lfWidth": "90",
										"lfFaceName": "宋体"
									}
								}
							]
						}
					]
				}
			]
		}
	}
}
`)
	navl01Viper.ReadConfig(bytes.NewBuffer(templete01))
	t.Logf("IsSet Programs:%+v", navl01Viper.IsSet("Programs"))
	t.Logf("Programs:%+v", navl01Viper.Get("Programs"))
	t.Logf("IsSet Programs.Program.Pages:%+v", navl01Viper.IsSet("Programs.Program.Pages"))
	t.Logf("Programs.Program.Pages:%+v", navl01Viper.Get("Programs.Program.Pages"))
	navl01Viper.Set("Programs.Program.Pages", "")

	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	sbuf, _ := json.MarshalToString(navl01Viper.AllSettings())
	t.Logf(`json:%+v`, sbuf)
}

func TestMultiLine0102(t *testing.T) {
	var json = jsoniter.ConfigCompatibleWithStandardLibrary
	var templete01 = []byte(`
	{
		"update":[
			{"nav_server_tag":"middle","who":{"x":0,"y":0},"text":"注意安全"}
		]
	}
`)

	arr := json.Get(templete01, "update")
	t.Logf("arr : %#v", arr)
	arr1 := json.Get(templete01, "update", 1)
	t.Logf("arr[1] : %#v err:%+v", arr1, arr1.LastError())
	t.Logf("arr[0].nav_server_tag : %+v", arr.Get(0).Get("nav_server_tag").ToString())

}

func TestHttpDownload(t *testing.T) {
	imgUrl := "https://www.twle.cn/static/i/img1.jpg"
	ctx := context.Background()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imgUrl, nil)
	if err != nil {
		t.Logf("ScreenShot err:%+v.", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Logf("ScreenShot err:%+v.", err)
	}
	defer resp.Body.Close()

	out, err := os.Create("img1.png")
	if err != nil {
		t.Logf("ScreenShot create file err:%+v", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		t.Logf("ScreenShot io.Copy err:%+v.", err)
	}

}

func TestJsonBuf(t *testing.T) {

}

func TestNavl1ProgramSt(t *testing.T) {
	t.Logf("Navl1_PrograNormal:%+v", gzb_ps.Navl1_PrograNormal)
	t.Logf("Navl1_PrograNormal_lines:%+v", gzb_ps.Navl1_PrograNormal_lines)
	t.Logf("Navl1_ProgramSpeed:%+v", gzb_ps.Navl1_ProgramSpeed)
	t.Logf("Navl1_ProgramSpeed_lines:%+v", gzb_ps.Navl1_ProgramSpeed_lines)
	t.Logf("Navl1_ProgramOverSpeed:%+v", gzb_ps.Navl1_ProgramOverSpeed)
	t.Logf("Navl1_ProgramOverSpeed_lines:%+v", gzb_ps.Navl1_ProgramOverSpeed_lines)
}

func TestNavl2ProgramSt(t *testing.T) {
	t.Logf("Navl2_PrograNormal:%+v", gzb_ps.Navl2_PrograNormal)
	t.Logf("Navl2_PrograNormal_lines:%+v", gzb_ps.Navl2_PrograNormal_lines)
	t.Logf("Navl2_ProgramSpeed:%+v", gzb_ps.Navl2_ProgramSpeed)
	t.Logf("Navl2_ProgramSpeed_lines:%+v", gzb_ps.Navl2_ProgramSpeed_lines)
	t.Logf("Navl2_ProgramOverSpeed:%+v", gzb_ps.Navl2_ProgramOverSpeed)
	t.Logf("Navl2_ProgramOverSpeed_lines:%+v", gzb_ps.Navl2_ProgramOverSpeed_lines)
}

func TestNavl3ProgramSt(t *testing.T) {
	t.Logf("Navl3_PrograNormal:%+v", gzb_ps.Navl3_PrograNormal)
	t.Logf("Navl3_PrograNormal_lines:%+v", gzb_ps.Navl3_PrograNormal_lines)
	t.Logf("Navl3_ProgramSpeed:%+v", gzb_ps.Navl3_ProgramSpeed)
	t.Logf("Navl3_ProgramSpeed_lines:%+v", gzb_ps.Navl3_ProgramSpeed_lines)
	t.Logf("Navl3_ProgramOverSpeed:%+v", gzb_ps.Navl3_ProgramOverSpeed)
	t.Logf("Navl3_ProgramOverSpeed_lines:%+v", gzb_ps.Navl3_ProgramOverSpeed_lines)
}

func TestFloatPrint(t *testing.T) {
	t.Logf("|%s|", fmt.Sprintf("%0.1f", 12.3124))
}

func TestJsonValid(t *testing.T) {
	str := `{\r\n  \"Programs\":{\r\n      \"Program\":{\r\n          \"Pages\":[\r\n              {\r\n                  \"Width\":\"288\",\r\n                  \"Height\":\"160\",\r\n                  \"Regions\":[\r\n                      {\r\n                          \"Layer\":1,\r\n                          \"Rect\":{\r\n                              \"X\":\"0\",\r\n                              \"Y\":\"0\",\r\n                              \"Width\":\"288\",\r\n                              \"Height\":\"160\"\r\n                          },\r\n                          \"Items\":[\r\n                              {\r\n                                  \"Type\":\"4\",\r\n                                  \"Text\":\"注意安全\",\r\n                                  \"TextColor\":\"#FFF0F5\",\r\n                                  \"IsScroll\":1,\r\n                                  \"Speed\":60,\r\n                                  \"CenteralAlign\":1,\r\n                                  \"LogFont\":{\r\n                                      \"lfHeight\":\"160\",\r\n                                      \"lfWeight\":\"160\",\r\n                                      \"lfItalic\":\"0\",\r\n                                      \"lfUnderline\":\"0\",\r\n                                      \"lfFaceName\":\"宋体\"\r\n                                  }\r\n                              }\r\n                          ]\r\n                      }\r\n                  ]\r\n              }\r\n          ]\r\n      }\r\n  }\r\n}`
	bs := ibyte.StringToBytes(str)
	bs = bytes.ReplaceAll(bs, []byte("\n"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\r"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\t"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\\n"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\\r"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\\t"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("\\"), []byte(""))
	bs = bytes.ReplaceAll(bs, []byte("  "), []byte(""))
	//bs = bytes.ReplaceAll(bs, []byte("\t\n\r"), []byte(""))
	//bs = bytes.ReplaceAll(bs, []byte("\\t\\n\\r\\"), []byte(""))
	// str = strings.ReplaceAll(str, "\\n", "")
	// str = strings.ReplaceAll(str, "\\r", "")
	// str = strings.ReplaceAll(str, "\\", "")
	// str = strings.ReplaceAll(str, "  ", "")
	t.Logf(`%s`, string(bs))
	if valid := jsoniter.Valid(bs); !valid {
		t.Logf("str is not valid json.")
	}

}

package feiyue_test

import (
	"encoding/hex"
	"fmt"
	"strings"
	"testing"

	feiyue "navigate/iot/drivers/tripodhead/feiyue"
)

func StringReplaceAll(ss *string) *string {
	*ss = strings.ReplaceAll(*ss, " ", "")
	*ss = strings.ReplaceAll(*ss, "\n", "")
	*ss = strings.ReplaceAll(*ss, "\t", "")
	return ss
}

func DeEncodetest(data []byte) {
	var pdu feiyue.PelcoDataUnit
	fmt.Println("----------------------")
	fmt.Printf("data: %0X \n", data)
	e := pdu.Decode(data)
	if e != nil {
		fmt.Printf("err: %v\n", e)
	}
	fmt.Printf("pdu: %s\n", pdu.String())

	sl, _ := pdu.Encode()
	fmt.Printf("buf: %0X \n", sl)
	fmt.Println("----------------------")
}

func wraperTest(t *testing.T, ss string) {
	ss = *StringReplaceAll(&ss)
	data, err := hex.DecodeString(ss)
	if err != nil {
		fmt.Printf("err: %+v", err)
	}
	DeEncodetest(data)
}

func TestDeEncodeCases(t *testing.T) {
	wraperTest(t, "FF 01 00 4B 82 14 E2")
	wraperTest(t, "FF 01 00 4B 0A 8C E2")
	wraperTest(t, "FF 01 00 4B 86 60 32")
	wraperTest(t, "FF 01 00 4B 06 40 92")
	wraperTest(t, "FF 01 00 05 00 77 7D")
	wraperTest(t, "FF 01 00 07 00 77 7F")
	wraperTest(t, "FF 01 00 51 00 00 52")
	wraperTest(t, "FF 01 00 59 00 00 5A")
	wraperTest(t, "FF 01 00 4B 23 28 97")
	wraperTest(t, "FF 01 00 59 23 28 A5")
	wraperTest(t, "FF 01 00 4B 8C A0 78")
	wraperTest(t, "FF 01 00 4D 8C A0 7A")
	wraperTest(t, "FF 01 00 5B 11 8C F9")
}

func BenchmarkEncode(b *testing.B) {
	var rdu feiyue.PelcoDataUnit
	ss := "FF 01 00 5B 11 8C F9"
	ss = *StringReplaceAll(&ss)
	data, err := hex.DecodeString(ss)
	if err != nil {
		b.Fatalf("err: %+v", err)
	}

	for i := 0; i < b.N; i++ {
		e := rdu.Decode(data)
		if e != nil {
			b.Fatalf("err: %v\n", e)
		}
	}
}

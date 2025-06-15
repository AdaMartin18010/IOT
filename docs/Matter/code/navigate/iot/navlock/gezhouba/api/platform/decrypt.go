package platform

import (
	"encoding/base64"
	"fmt"
	"sync"

	cm "navigate/common/model"
	bs "navigate/internal/bytestr"

	openssl "github.com/forgoer/openssl"
)

var (
	aesKey = []byte("lB2BxrJdI4UUjK3KEZyQ0obuSgavB1SYJuAFq9oVw0Y=")
	aesIv  = []byte("6lra6ceX26Fazwj1R4PCOg==")
	once   sync.Once
	L      = cm.L
)

const (
	TokenKey = "15A03ABB189F367A"
	TokenID  = "9910A487188F1FB7"
)

func init() {
	once.Do(func() {
		if err := decodeKV(); err != nil {
			L.Sugar().Errorf("DecodeKV error: %+v", err)
		}
	})
}

func decodeKV() error {
	var err error
	aesKey, err = Base64Decode(aesKey)
	if err != nil {
		return fmt.Errorf("Base64Decode err is : %+v", err)
	}

	aesIv, err = Base64Decode(aesIv)
	if err != nil {
		return fmt.Errorf("Base64Decode err is : %+v", err)
	}

	return nil
}

// 加密解密要点：
//
//	1、处理数据,使用k,v;k,v的使用先要base64解码后,加密解密http的消息包 AES PKCS7和PKCS5 是一样的.
//	2、对数据进行加密,采用AES加密方法中CBC加密模式.
//	3、对加密后的数据,进行base64编码再发送;解密先是针对base64解码后再解密
func Base64Decode(src []byte) (dst []byte, err error) {
	dst, err = base64.StdEncoding.DecodeString(bs.BytesToString(src))
	if err != nil {
		return nil, fmt.Errorf("base64.StdEncoding.DecodeString key err is : %+v", err)
	}
	return
}

func Base64Encode(src []byte) (dst []byte) {
	dst = bs.StringToBytes(base64.StdEncoding.EncodeToString(src))
	return
}

// 1.加密-AesCBC 2.base64编码
func EncryptEncodeWraper(src []byte) (dst []byte, err error) {
	dst, err = openssl.AesCBCEncrypt(src, aesKey, aesIv, openssl.PKCS7_PADDING)
	if err != nil {
		return nil, err
	}

	dst = bs.StringToBytes(base64.StdEncoding.EncodeToString(dst))

	return
}

// 1.base64解码 2.解密-AesCBC
func DecodeDecryptWraper(src []byte) (dst []byte, err error) {
	dst, err = Base64Decode(src)
	if err != nil {
		return
	}

	dst, err = openssl.AesCBCDecrypt(dst, aesKey, aesIv, openssl.PKCS7_PADDING)
	if err != nil {
		return nil, err
	}

	return
}

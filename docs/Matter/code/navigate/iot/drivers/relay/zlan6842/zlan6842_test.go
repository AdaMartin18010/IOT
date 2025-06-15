package zlan6842

import (
	"errors"
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"github.com/goburrow/modbus"
)

func Test(t *testing.T) {
	// Modbus TCP
	handler := modbus.NewTCPClientHandler("172.60.223.10:502")
	handler.Timeout = 2 * time.Second
	handler.SlaveId = 0x01
	handler.Logger = log.New(os.Stdout, "zlan-test: ", log.LstdFlags)
	// Connect manually so that multiple requests are handled in one connection session
	err := handler.Connect()
	defer handler.Close()
	if err != nil {
		log.Fatalf("err : %v\n", err)
	}
	client := modbus.NewClient(handler)

	for {
		{ //测试打开关闭DO
			var indexOfDO, switchOnOff uint16
			indexOfDO = 1
			switchOnOff = 0xFF00
			for ; indexOfDO <= 8; indexOfDO++ {
				iDO := 0x0010 + indexOfDO - 1

				// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
				// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
				_, err := client.WriteSingleCoil(iDO, switchOnOff)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}

				// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
				results, err := client.ReadCoils(0x0010, 0x0008)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//屏蔽掉其他位
				bit := 0x01 << (indexOfDO - 1)
				res := results[0] & byte(bit)
				//判断当前位
				res = res & ^byte(bit)

				if res != 0x00 {
					log.Printf("err *****************  [on]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
				}

			}

			indexOfDO = 1
			switchOnOff = 0x0000
			for ; indexOfDO <= 8; indexOfDO++ {
				iDO := 0x0010 + indexOfDO - 1

				// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
				// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
				_, err := client.WriteSingleCoil(iDO, switchOnOff)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//log.Printf("result: %v\n", results)

				// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
				results, err := client.ReadCoils(0x0010, 0x0008)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//屏蔽掉其他位
				bit := 0x01 << (indexOfDO - 1)
				res := results[0] & byte(bit)

				if res != 0x00 {
					log.Printf("err ***************** [off]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
				}

			}

		}

		{
			// 测试读取DI AI
			// 读取DI:{1-8}对应的状态一字节,低-高位对应 cmd=0x02
			results, err := client.ReadDiscreteInputs(0x0000, 0x0008)
			if err != nil {
				log.Fatalf("err : %v\n", err)
			}
			if l := len(results); l > 0 {
				//log.Printf("result: %b\n", results)
				ss := ""
				for i := l - 1; i >= 0; i-- {
					for j := 7; j >= 0; j-- {
						bits := 0x01 << j
						val := results[i] & byte(bits)
						if val == 0x00 {
							ss += fmt.Sprintf("DI[%d]:%s \t", i*8+j+1, "OFF")
						} else {
							ss += fmt.Sprintf("DI[%d]:%s \t", i*8+j+1, "ON")
						}

					}

				}
				log.Printf("result: %b, ss:%s\n", results, ss)
			}

			// 读取AI:{1-8}对应的状态一字节,低-高位对应;从第0x00(AI1)开始读取0x00008个的数值
			// int16的数值对应电流大小  cmd=0x04
			results, err = client.ReadInputRegisters(0x0000, 0x0008)
			if err != nil {
				log.Fatalf("err : %v\n", err)
			}
			if l := len(results); l > 0 {
				//log.Printf("result: %b\n", results)
				ss := ""
				var val uint16
				var f float32
				for j, i := 0, l-1; i >= 0; i -= 2 {
					val = uint16(results[j]) << 8
					val = val | uint16(results[j+1])
					f = float32(val) * 5 / 1024
					j += 2
					ss += fmt.Sprintf("AI[%d]:%.3f-%X \t", j/2, f, val)
				}
				log.Printf("result: %X, ss:%s\n", results, ss)
			}
		}
		//os.Exit(0)
		//time.Sleep(20 * time.Microsecond)
	}

}

func SetDOs(cl modbus.Client, dos DOitems) (e error) {
	if len(dos) <= 0 {
		return errors.New("DOitems length <=0")
	}
	var switchOnOff uint16
	for _, do := range dos {
		if do.Id > ZlanDOitemsMax || do.Id < ZlanDOindex {
			return fmt.Errorf("DOitems ID:(no valid) %v", do)
		}
		iDO := ZlanDOaddr + do.Id - 1
		// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
		// 设置开关:{on:0xFF00,off:0x0000}
		if do.On {
			switchOnOff = ZlanSwitchOn
		} else {
			switchOnOff = ZlanSwitchOff
		}

		//cmd=0x05
		_, err := cl.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			return err
		}
	}
	return
}

func GetDOs(cl modbus.Client) (dosStatus DOitems, e error) {
	// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
	results, err := cl.ReadCoils(ZlanDOaddr, ZlanDOitemsMax)
	if err != nil {
		return nil, err
	}

	dosStatus = make(DOitems, ZlanDOitemsMax)
	var indexOfDO uint16
	// DO id begin with 1+
	for indexOfDO = 1; indexOfDO <= ZlanDOitemsMax; indexOfDO++ {
		dosStatus[indexOfDO-1].Id = indexOfDO
		//屏蔽掉其他位
		bit := 0x01 << (indexOfDO - 1)
		res := results[0] & byte(bit)
		//判断当前位
		res = res & byte(bit)
		if res != 0x00 {
			dosStatus[indexOfDO-1].On = true
		} else {
			dosStatus[indexOfDO-1].On = false
		}
	}
	return
}

func SetDOsAllOn(cl modbus.Client) (cll modbus.Client) {
	var indexOfDO, switchOnOff uint16
	indexOfDO = 1
	switchOnOff = ZlanSwitchOn
	for ; indexOfDO <= ZlanDOitemsMax; indexOfDO++ {
		iDO := 0x0010 + indexOfDO - 1

		// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
		// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
		_, err := cl.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			log.Fatalf("err : %v\n", err)
		}

		/*  //校验通讯协议使用
		// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
		results, err := cl.ReadCoils(0x0010, 0x0008)
		if err != nil {
			log.Fatalf("err : %v\n", err)
		}
		//屏蔽掉其他位
		bit := 0x01 << (indexOfDO - 1)
		res := results[0] & byte(bit)
		//判断当前位
		res = res & ^byte(bit)
		if res != 0x00 {
			log.Printf("err * [on]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
		}
		*/
	}
	cll = cl
	return
}

func SetDOsAllOff(cl modbus.Client) (cll modbus.Client) {
	var indexOfDO, switchOnOff uint16
	indexOfDO = 1
	switchOnOff = ZlanSwitchOff
	for ; indexOfDO <= ZlanDOitemsMax; indexOfDO++ {
		iDO := 0x0010 + indexOfDO - 1

		// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
		// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
		_, err := cl.WriteSingleCoil(iDO, switchOnOff)
		if err != nil {
			log.Fatalf("err : %v\n", err)
		}

		/* //校验通信协议使用
		// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
		results, err := cl.ReadCoils(0x0010, zlanDOitemsMax)
		if err != nil {
			log.Fatalf("err : %v\n", err)
		}
		//屏蔽掉其他位
		bit := 0x01 << (indexOfDO - 1)
		res := results[0] & byte(bit)
		if res != 0x00 {
			log.Printf("err ** [off]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
		}
		*/
	}

	return
}

func Test2(t *testing.T) {
	// Modbus TCP
	handler := modbus.NewTCPClientHandler("172.60.223.10:502")
	handler.Timeout = 2 * time.Second
	handler.SlaveId = 0x01
	handler.Logger = log.New(os.Stdout, "zlan-test: ", log.LstdFlags)
	// Connect manually so that multiple requests are handled in one connection session
	err := handler.Connect()
	defer handler.Close()
	if err != nil {
		log.Fatalf("err : %v\n", err)
	}
	client := modbus.NewClient(handler)

	for {
		{ //测试打开关闭DO
			var indexOfDO, switchOnOff uint16
			indexOfDO = 1
			switchOnOff = 0xFF00
			for ; indexOfDO <= 8; indexOfDO++ {
				iDO := 0x0010 + indexOfDO - 1

				// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
				// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
				_, err := client.WriteSingleCoil(iDO, switchOnOff)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}

				// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
				results, err := client.ReadCoils(0x0010, 0x0008)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//屏蔽掉其他位
				bit := 0x01 << (indexOfDO - 1)
				res := results[0] & byte(bit)
				//判断当前位
				res = res & ^byte(bit)

				if res != 0x00 {
					log.Printf("err *****************  [on]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
				}

			}

			indexOfDO = 1
			switchOnOff = 0x0000
			for ; indexOfDO <= 8; indexOfDO++ {
				iDO := 0x0010 + indexOfDO - 1

				// 设置DO:{1-8}对应地址{DO1:0x0010--DO8:0x0017}
				// 设置开关:{on:0xFF00,off:0x0000} cmd=0x05
				_, err := client.WriteSingleCoil(iDO, switchOnOff)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//log.Printf("result: %v\n", results)

				// 读取DO:{1-8}对应的状态一字节,低-高位对应 cmd=0x01
				results, err := client.ReadCoils(0x0010, 0x0008)
				if err != nil {
					log.Fatalf("err : %v\n", err)
				}
				//屏蔽掉其他位
				bit := 0x01 << (indexOfDO - 1)
				res := results[0] & byte(bit)

				if res != 0x00 {
					log.Printf("err ***************** [off]:res : %b ; indexOfDO : %x\n", res, indexOfDO)
				}

			}

		}

		{
			//测试读取DI AI

			// 读取DI:{1-8}对应的状态一字节,低-高位对应 cmd=0x02
			//
			results, err := client.ReadDiscreteInputs(0x0000, 0x0008)
			if err != nil {
				log.Fatalf("err : %v\n", err)
			}
			if l := len(results); l > 0 {
				//log.Printf("result: %b\n", results)
				ss := ""
				for i := l - 1; i >= 0; i-- {
					for j := 7; j >= 0; j-- {
						bits := 0x01 << j
						val := results[i] & byte(bits)
						if val == 0x00 {
							ss += fmt.Sprintf("DI[%d]:%s \t", i*8+j+1, "OFF")
						} else {
							ss += fmt.Sprintf("DI[%d]:%s \t", i*8+j+1, "ON")
						}

					}

				}
				log.Printf("result: %b, ss:%s\n", results, ss)
			}

			// 读取AI:{1-8}对应的状态一字节,低-高位对应;从第0x00(AI1)开始读取0x00008个的数值
			// int16的数值对应电流大小  cmd=0x04
			results, err = client.ReadInputRegisters(0x0000, 0x0008)
			if err != nil {
				log.Fatalf("err : %v\n", err)
			}
			if l := len(results); l > 0 {
				//log.Printf("result: %b\n", results)
				ss := ""
				var val uint16
				var f float32
				for j, i := 0, l-1; i >= 0; i -= 2 {
					val = uint16(results[j]) << 8
					val = val | uint16(results[j+1])
					f = float32(val) * 5 / 1024
					j += 2
					ss += fmt.Sprintf("AI[%d]:%.3f-%X \t", j/2, f, val)
				}
				log.Printf("result: %X, ss:%s\n", results, ss)
			}
		}
		//os.Exit(0)
		//time.Sleep(20 * time.Microsecond)
	}

}

-- MySQL dump 10.13  Distrib 8.0.30, for Win64 (x86_64)
--
-- Host: localhost    Database: gvaweb
-- ------------------------------------------------------
-- Server version	8.0.30

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `nav_animation_history`
--

DROP TABLE IF EXISTS `nav_animation_history`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_animation_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `createdAt` datetime DEFAULT NULL COMMENT '发生时间',
  `navLockId` int DEFAULT NULL COMMENT '船闸名称',
  `scheduleId` varchar(255) DEFAULT NULL COMMENT '闸次号',
  `navLockStatus` int DEFAULT NULL COMMENT '0: "闸室运行状态未知",1: "上行进闸中",2: "上行进闸完毕",3: "上行出闸中",4: "上行出闸完毕",5: "下行进闸中",\r\n6: "下行进闸完毕",7: "下行出闸中",8: "下行出闸完毕"',
  `gateLeftUpStatus` int DEFAULT NULL COMMENT '左上闸门状态0: "闸门状态未知",1: "闸门开运行",2: "闸门开终",3: "闸门关运行",4: "闸门关终",',
  `gateLeftDownStatus` int DEFAULT NULL COMMENT '左下闸门状态0: "闸门状态未知",1: "闸门开运行",2: "闸门开终",3: "闸门关运行",4: "闸门关终",',
  `gateRightDownStatus` int DEFAULT NULL COMMENT '左下闸门状态0: "闸门状态未知",1: "闸门开运行",2: "闸门开终",3: "闸门关运行",4: "闸门关终",',
  `gateRightUpStatus` int DEFAULT NULL COMMENT '左下闸门状态0: "闸门状态未知",1: "闸门开运行",2: "闸门开终",3: "闸门关运行",4: "闸门关终",',
  `stoplineUpStatus` int DEFAULT NULL COMMENT '禁停状态（正常，失联）',
  `stoplineUpWidth` int DEFAULT NULL COMMENT '禁停线上游宽度',
  `stoplineUpWarn` int DEFAULT NULL COMMENT '禁停线上游告警（0无，1，告警，2警报）',
  `stoplineUpDistance` double DEFAULT NULL COMMENT '上游超出禁停线距离',
  `stoplineDownStatus` int DEFAULT NULL COMMENT '禁停状态（1正常，0失联）',
  `stoplineDownWidth` int DEFAULT NULL COMMENT '禁停线下游宽度',
  `stoplineDownWarn` int DEFAULT NULL COMMENT '禁停线下游告警（0无，1，告警，2警报）',
  `stoplineDownDistance` double DEFAULT NULL COMMENT '下游超出禁停线距离',
  `shipLeftDistance` double DEFAULT NULL COMMENT '左船雷达距离',
  `shipLeftSpeed` double DEFAULT NULL COMMENT '左船雷达速度',
  `shipLeftWarn` int DEFAULT NULL COMMENT '左船告警（0无，1，超速）',
  `shipRightDistance` double DEFAULT NULL COMMENT '右船雷达距离',
  `shipRightSpeed` double DEFAULT NULL COMMENT '右船雷达速度',
  `shipRightWarn` int DEFAULT NULL COMMENT '右船告警（0无，1，超速）',
  `selfLefeUpState` int DEFAULT NULL COMMENT '左上自有设备状态0：正常； 1：开关量故障； 2：云台故障； 3：雷达故障； 4：开关量和云台故障； 5：开关量和雷达故障； 6：云台和雷达故障；7：全故障',
  `selfLefeDownState` int DEFAULT NULL COMMENT '左下自有设备状态0：正常； 1：开关量故障； 2：云台故障； 3：雷达故障； 4：开关量和云台故障； 5：开关量和雷达故障； 6：云台和雷达故障 ；7：全故障',
  `selfRightUpState` int DEFAULT NULL COMMENT '右上自有设备状态0：正常； 1：开关量故障； 2：云台故障； 3：雷达故障； 4：开关量和云台故障； 5：开关量和雷达故障； 6：云台和雷达故障；7：全故障',
  `selfRightDownState` int DEFAULT NULL COMMENT '右下自有设备状态0：正常； 1：开关量故障； 2：云台故障； 3：雷达故障； 4：开关量和云台故障； 5：开关量和雷达故障； 6：云台和雷达故障；7：全故障',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `nav_animation_history`
--

LOCK TABLES `nav_animation_history` WRITE;
/*!40000 ALTER TABLE `nav_animation_history` DISABLE KEYS */;
/*!40000 ALTER TABLE `nav_animation_history` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-10-03 23:02:58

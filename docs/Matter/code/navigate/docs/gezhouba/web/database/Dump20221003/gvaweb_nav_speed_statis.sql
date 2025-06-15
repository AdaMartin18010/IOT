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
-- Table structure for table `nav_speed_statis`
--

DROP TABLE IF EXISTS `nav_speed_statis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_speed_statis` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `scheduleStartTime` datetime NOT NULL COMMENT '闸次开始时间',
  `navLockId` int DEFAULT NULL COMMENT '船闸名称',
  `scheduleId` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '闸次号',
  `overSpeedCount` bigint DEFAULT NULL COMMENT '超速次',
  `upInCount` bigint DEFAULT NULL COMMENT '上行进闸超速次',
  `upOutCount` bigint DEFAULT NULL COMMENT '上行出闸超速次',
  `downInCount` bigint DEFAULT NULL COMMENT '下行进闸超速次',
  `downOutCount` bigint DEFAULT NULL COMMENT '下行出闸超速次',
  `maxSpeed` double DEFAULT NULL COMMENT '最高速度',
  `upInSpeed` double DEFAULT NULL COMMENT '上行进闸平均速度',
  `upOutSpeed` double DEFAULT NULL COMMENT '上行出闸平均速度',
  `downInSpeed` double DEFAULT NULL COMMENT '下行进闸平均速度',
  `downOutSpeed` double DEFAULT NULL COMMENT '下行出闸平均速度',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `nav_speed_statis`
--

LOCK TABLES `nav_speed_statis` WRITE;
/*!40000 ALTER TABLE `nav_speed_statis` DISABLE KEYS */;
INSERT INTO `nav_speed_statis` VALUES (1,'2022-09-19 16:01:40',1,'20220920-1',20,10,5,2,3,2.1,1.6,2.3,2.1,2),(2,'2022-09-19 16:01:40',2,'20220920-1',20,10,5,2,3,2.1,1.6,2.3,2.1,2),(3,'2022-09-19 16:01:40',3,'20220920-1',20,10,5,2,3,2.1,1.6,2.3,2.1,2);
/*!40000 ALTER TABLE `nav_speed_statis` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-10-03 23:02:57

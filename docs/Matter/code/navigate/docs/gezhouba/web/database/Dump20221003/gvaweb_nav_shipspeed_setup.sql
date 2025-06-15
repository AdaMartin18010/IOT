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
-- Table structure for table `nav_shipspeed_setup`
--

DROP TABLE IF EXISTS `nav_shipspeed_setup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_shipspeed_setup` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `navLockId` bigint DEFAULT NULL COMMENT '船闸名称',
  `distance` double DEFAULT NULL COMMENT '距离闸门长度',
  `speedMax` double DEFAULT NULL COMMENT '最大速度',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `index_navdis` (`navLockId`,`distance`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `nav_shipspeed_setup`
--

LOCK TABLES `nav_shipspeed_setup` WRITE;
/*!40000 ALTER TABLE `nav_shipspeed_setup` DISABLE KEYS */;
INSERT INTO `nav_shipspeed_setup` VALUES (2,1,50,2.4),(3,1,100,1.7),(4,1,150,2),(9,2,50,2),(10,2,100,2),(11,2,150,2),(12,2,200,2),(13,2,280,2),(14,3,30,2),(15,3,50,2),(16,3,100,2),(17,3,150,2),(18,3,180,3),(23,1,200,2.4),(24,1,350,2);
/*!40000 ALTER TABLE `nav_shipspeed_setup` ENABLE KEYS */;
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

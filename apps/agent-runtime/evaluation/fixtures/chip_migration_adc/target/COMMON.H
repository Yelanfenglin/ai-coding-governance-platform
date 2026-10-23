/******************* (C) COPYRIGHT 2014~2025 MASSES CHIP ************************
* File Name          : MassesChip_Define.h
* Author             : MASSES CHIP
* Version            : V1.03
* Date               : 2025/08/15
********************************************************************************
 


*******************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "zc.h"
#include "MASSESCHIP_DEFINE.H"


bit	b_TC1_Flag;

void SYSTEM_INIT(void);					//系统初始化
void TC1_INIT(void);					//TC1初始化
void TC1_Interrupt (void);				//TC1中断处理函数
void TC1_Work(void);					//TC1处理函数


#endif

/*********************************END OF FILE************************************/
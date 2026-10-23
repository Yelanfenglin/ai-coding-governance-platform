/* System initialization from M9P6362/M9P6362_Demo_V1.07.zip, PWM0/main.C; MASSES CHIP. */
#include "demo_platform.h"

void Demo_PlatformInit(void)
{ 
//----------------------------------------------------
//Init 系统时钟
	OSCM = 0x00;
//----------------------------------------------------	            
//Init IO
	//IOA
	ANSA = 0x00;               //0:数字口	1:模拟口 
	PUA = 0x00;                //0:上拉关闭 1:上拉使能
	PDA = 0x00;				   //0:下拉关闭 1:下拉使能
	IOA = 0x00;                //IO口数据
	OEA = 0xFF;                //0:输入 1:输出
	//IOB
	ANSB = 0x00;  
	PUB = 0x00;                
	PDB = 0x00;				   
	IOB = 0x00;                
	OEB = 0xFF;                
	//IOC
	ANSC = 0x00;  
	PUC = 0x00;               
	PDC = 0x00;				   
	IOC = 0x00;                
	OEC = 0xFF;
	//IOD
	ANSD = 0x00;  
	PUD = 0x00;               
	PDD = 0x00;				   
	IOD = 0x00;                
	OED = 0xFF;
//----------------------------------------------------     
//Init Interrupt
	GIEH = 0;                  //1:全局中断高优先级中断使能,0:屏蔽全局中断高优先级中断
	GIEL = 0;                  //1:全局中断低优先级中断使能,0:屏蔽全局中断低优先级中断
}

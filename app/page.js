"use client";

import Image from "next/image";
import { useEffect } from "react";

export default function Home() {
  useEffect(() => {
    const eventSource = new EventSource('http://localhost:8000/read_dataset');
    eventSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const container = document.getElementById('traffic-counter');
      const newElement = document.createElement('div');
      newElement.textContent = data.data;
      container.insertBefore(newElement, container.firstChild);
      
      // 删除超出窗口的内容
      while (container.children.length > 20) {
        container.removeChild(container.lastChild);
      }
    };
    return () => eventSource.close();
  }, []);
  return (
    <div className="flex flex-col md:flex-row h-screen p-6 gap-6 bg-gray-50">
      {/* 左侧车辆流量 */}
      <div className="md:w-1/3 bg-white rounded-xl shadow-sm p-6 flex flex-col transition-all duration-300 hover:shadow-md">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <h2 className="text-xl font-semibold text-gray-800">车辆流量实时监控</h2>
        </div>
        
        <div className="flex-1 bg-gray-50 rounded-lg p-4 flex items-center justify-center"  id="traffic-counter">
        </div>
      </div>

      {/* 右侧内容容器 */}
      <div className="md:w-2/3 flex flex-col gap-6">
        {/* 攻击类型按钮组 - 现代胶囊按钮设计 */}
        <div className="flex flex-wrap gap-3">
          {['DoS攻击', '重放攻击', '欺骗攻击', '中间人攻击', '模糊测试'].map((type) => (
            <button 
              key={type}
              className="px-4 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 
              text-white text-sm font-medium rounded-full shadow-sm hover:shadow 
              transform hover:-translate-y-0.5 transition-all duration-200"
            >
              {type}
            </button>
          ))}
        </div>

        {/* 检测数据展示区 - 现代仪表盘设计 */}
        <div className="flex-1 bg-white rounded-xl shadow-sm p-6 transition-all duration-300 hover:shadow-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">安全检测数据</h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: '异常流量', value: '0', color: 'red' },
              { label: '可疑连接', value: '0', color: 'yellow' },
              { label: '阻断事件', value: '0', color: 'blue' },
              { label: '系统状态', value: '正常', color: 'green' }
            ].map((item, i) => (
              <div 
                key={i} 
                className="bg-gray-50 p-4 rounded-lg border-l-4 border-l-blue-500"
              >
                <div className="text-sm text-gray-500 mb-1">{item.label}</div>
                <div className="text-2xl font-bold text-gray-800">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

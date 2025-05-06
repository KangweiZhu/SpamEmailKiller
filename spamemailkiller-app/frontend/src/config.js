// 判断是否在开发环境
const isDev = process.env.NODE_ENV === 'development';

// API基础URL
export const API_BASE_URL = isDev 
    ? 'http://localhost:5002'  // 开发环境
    : 'http://localhost:5002'; // 生产环境（与后端在同一台机器上）

// 其他配置
export const APP_CONFIG = {
    name: 'Spam Killer',
    version: '1.0.0',
    description: 'Email Spam Detection Application'
}; 
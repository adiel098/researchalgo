// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://haleviadiel.csariel.xyz/api' // שרת הייצור
  : '/api'; // שרת מקומי

export { API_BASE_URL };

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

// Check if running in demo mode
export function isDemoMode() {
  return localStorage.getItem('token') === 'demo-token'
}

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 responses — but NOT in demo mode
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !isDemoMode()) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api

import { useState, useEffect } from 'react'
import { Activity, Shield, Brain, Database } from 'lucide-react'
import api, { isDemoMode } from '../api'
import { MOCK_HEALTH } from '../demo'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const user = JSON.parse(localStorage.getItem('user') || '{}')

  useEffect(() => {
    if (isDemoMode()) {
      setHealth(MOCK_HEALTH)
      return
    }
    api.get('/health').then(({ data }) => {
      if (data.success) setHealth(data.data)
    }).catch(() => {})
  }, [])

  const services = health?.services || {}

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Welcome back, <span className="text-gray-700">{user.email}</span>
        </p>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatusCard
          icon={Activity}
          label="API"
          status={services.api || 'unknown'}
        />
        <StatusCard
          icon={Database}
          label="PostgreSQL"
          status={services.postgres || 'unknown'}
        />
        <StatusCard
          icon={Database}
          label="Qdrant"
          status={services.qdrant || 'unknown'}
        />
        <StatusCard
          icon={Shield}
          label="System"
          status={health?.status || 'unknown'}
        />
      </div>

      {/* Features */}
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Features</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FeatureCard
          icon={Brain}
          title="Intent Classification"
          description="RoBERTa classifies queries into 6 categories: factual, person, time, location, explanation, other"
        />
        <FeatureCard
          icon={Database}
          title="Targeted Retrieval"
          description="Each intent routes to a dedicated Qdrant collection, filtered by tenant_id for isolation"
        />
        <FeatureCard
          icon={Shield}
          title="Hallucination Detection"
          description="3-method check: retrieval confidence + entropy analysis + LLM verification"
        />
        <FeatureCard
          icon={Activity}
          title="Self-Healing"
          description="Auto-retries with corrective strategies: expand retrieval, strict grounding, broaden search"
        />
      </div>

      {/* Quick Info */}
      <div className="mt-8 bg-gray-50 rounded-xl p-6 border border-gray-200">
        <h3 className="font-medium text-gray-700 mb-3">Your Account</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Role:</span>{' '}
            <span className="font-medium capitalize">{user.role}</span>
          </div>
          <div>
            <span className="text-gray-500">Tenant:</span>{' '}
            <span className="font-medium">{user.tenant_name}</span>
          </div>
          <div>
            <span className="text-gray-500">Version:</span>{' '}
            <span className="font-medium">{health?.version || '2.0.0'}</span>
          </div>
          <div>
            <span className="text-gray-500">Permissions:</span>{' '}
            <span className="font-medium">
              {user.role === 'admin' ? 'Full access' : user.role === 'writer' ? 'Query + Ingest' : 'Query only'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ icon: Icon, label, status }) {
  const color = status === 'up' || status === 'healthy'
    ? 'text-green-600 bg-green-50 border-green-200'
    : status === 'degraded'
    ? 'text-amber-600 bg-amber-50 border-amber-200'
    : 'text-red-600 bg-red-50 border-red-200'

  return (
    <div className={`rounded-xl border p-4 ${color}`}>
      <Icon size={20} className="mb-2" />
      <div className="text-xs font-medium opacity-70">{label}</div>
      <div className="text-sm font-semibold capitalize">{status}</div>
    </div>
  )
}

function FeatureCard({ icon: Icon, title, description }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <Icon size={20} className="text-indigo-600 mb-3" />
      <h3 className="font-medium text-gray-900 text-sm">{title}</h3>
      <p className="text-xs text-gray-500 mt-1 leading-relaxed">{description}</p>
    </div>
  )
}

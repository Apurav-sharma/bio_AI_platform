'use client'

import React, { useState } from 'react';
import { AlertCircle, Activity, User, TrendingUp, Clock, Shield } from 'lucide-react';

const AegisCancerPrediction = () => {
  const [formData, setFormData] = useState({
    TP53: '',
    BRCA1: '',
    EGFR: '',
    MYC: '',
    age: '',
    bmi: '',
    smoking_history: '',
    family_history: '',
    previous_cancer_history: '',
    inflammatory_markers: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  const fieldDescriptions = {
    TP53: "Tumor suppressor gene expression level (0-15)",
    BRCA1: "DNA repair gene expression level (0-15)",
    EGFR: "Growth factor receptor expression level (0-15)",
    MYC: "Oncogene expression level (0-15)",
    age: "Patient age in years (18-120)",
    bmi: "Body Mass Index (12-50)",
    smoking_history: "Smoking history (0=No, 1=Yes)",
    family_history: "Family cancer history (0=No, 1=Yes)",
    previous_cancer_history: "Previous cancer diagnosis (0=No, 1=Yes)",
    inflammatory_markers: "Inflammatory markers level (0-20)"
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const loadExampleData = (type) => {
    const examples = {
      high_risk: {
        TP53: '2.5',
        BRCA1: '3.0',
        EGFR: '8.5',
        MYC: '9.0',
        age: '65',
        bmi: '32.0',
        smoking_history: '1',
        family_history: '1',
        previous_cancer_history: '1',
        inflammatory_markers: '8.5'
      },
      low_risk: {
        TP53: '6.0',
        BRCA1: '5.5',
        EGFR: '4.0',
        MYC: '4.5',
        age: '35',
        bmi: '22.0',
        smoking_history: '0',
        family_history: '0',
        previous_cancer_history: '0',
        inflammatory_markers: '1.0'
      }
    };
    setFormData(examples[type]);
    setPrediction(null);
    setError('');
  };

  const validateForm = () => {
    const requiredFields = Object.keys(formData);
    const emptyFields = requiredFields.filter(field => !formData[field]);

    if (emptyFields.length > 0) {
      setError(`Please fill in all fields. Missing: ${emptyFields.join(', ')}`);
      return false;
    }

    // Validate ranges
    const validations = {
      TP53: { min: 0, max: 15 },
      BRCA1: { min: 0, max: 15 },
      EGFR: { min: 0, max: 15 },
      MYC: { min: 0, max: 15 },
      age: { min: 18, max: 120 },
      bmi: { min: 12, max: 50 },
      smoking_history: { min: 0, max: 1 },
      family_history: { min: 0, max: 1 },
      previous_cancer_history: { min: 0, max: 1 },
      inflammatory_markers: { min: 0, max: 20 }
    };

    for (const [field, range] of Object.entries(validations)) {
      const value = parseFloat(formData[field]);
      if (isNaN(value) || value < range.min || value > range.max) {
        setError(`${field}: Value must be between ${range.min} and ${range.max}`);
        return false;
      }
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const payload = {};
      Object.keys(formData).forEach(key => {
        payload[key] = parseFloat(formData[key]);
      });

      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (percentage) => {
    if (percentage < 25) return 'text-green-600 bg-green-50 border-green-200';
    if (percentage < 50) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (percentage < 75) return 'text-orange-600 bg-orange-50 border-orange-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getRiskIcon = (percentage) => {
    if (percentage < 25) return <Shield className="w-6 h-6" />;
    if (percentage < 50) return <AlertCircle className="w-6 h-6" />;
    if (percentage < 75) return <TrendingUp className="w-6 h-6" />;
    return <AlertCircle className="w-6 h-6" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Activity className="w-10 h-10 text-indigo-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">AEGIS</h1>
          </div>
          <p className="text-xl text-gray-600 mb-2">AI-Enhanced Genomic Intelligence System</p>
          <p className="text-gray-500">Cancer Risk Prediction & Survival Analysis</p>
        </div>

        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center mb-6">
              <User className="w-6 h-6 text-indigo-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-900">Patient Information</h2>
            </div>

            {/* API URL Configuration */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                API URL
              </label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="http://localhost:8000"
              />
            </div>

            {/* Example Data Buttons */}
            <div className="mb-6 flex gap-3">
              <button
                type="button"
                onClick={() => loadExampleData('high_risk')}
                className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors text-sm font-medium"
              >
                Load High Risk Example
              </button>
              <button
                type="button"
                onClick={() => loadExampleData('low_risk')}
                className="px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors text-sm font-medium"
              >
                Load Low Risk Example
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Genetic Biomarkers */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
                  Genetic Biomarkers
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {['TP53', 'BRCA1', 'EGFR', 'MYC'].map((field) => (
                    <div key={field}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field}
                      </label>
                      <input
                        type="number"
                        name={field}
                        value={formData[field]}
                        onChange={handleInputChange}
                        step="0.1"
                        min="0"
                        max="15"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                        title={fieldDescriptions[field]}
                      />
                      <p className="text-xs text-gray-500 mt-1">{fieldDescriptions[field]}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Clinical Factors */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                  Clinical Factors
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleInputChange}
                      min="18"
                      max="120"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">BMI</label>
                    <input
                      type="number"
                      name="bmi"
                      value={formData.bmi}
                      onChange={handleInputChange}
                      step="0.1"
                      min="12"
                      max="50"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Smoking History</label>
                    <select
                      name="smoking_history"
                      value={formData.smoking_history}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      <option value="">Select</option>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Family History</label>
                    <select
                      name="family_history"
                      value={formData.family_history}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      <option value="">Select</option>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Previous Cancer</label>
                    <select
                      name="previous_cancer_history"
                      value={formData.previous_cancer_history}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      <option value="">Select</option>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Inflammatory Markers</label>
                    <input
                      type="number"
                      name="inflammatory_markers"
                      value={formData.inflammatory_markers}
                      onChange={handleInputChange}
                      step="0.1"
                      min="0"
                      max="20"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <span className="text-red-700">{error}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  'Predict Cancer Risk'
                )}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center mb-6">
              <TrendingUp className="w-6 h-6 text-indigo-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-900">Prediction Results</h2>
            </div>

            {!prediction && !loading && (
              <div className="text-center py-12">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Enter patient information and click `Predict Cancer Risk` to see results</p>
              </div>
            )}

            {loading && (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Analyzing patient data...</p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6">
                {/* Risk Assessment */}
                <div className={`p-6 rounded-xl border-2 ${getRiskColor(prediction.cancer_risk_percentage)}`}>
                  <div className="flex items-center mb-3">
                    {getRiskIcon(prediction.cancer_risk_percentage)}
                    <h3 className="text-xl font-semibold ml-3">{prediction.risk_category}</h3>
                  </div>
                  <div className="text-3xl font-bold mb-2">
                    {prediction.cancer_risk_percentage}%
                  </div>
                  <p className="font-medium">Cancer Risk Probability</p>
                </div>

                {/* Survival Prediction */}
                <div className="bg-blue-50 border-2 border-blue-200 p-6 rounded-xl">
                  <div className="flex items-center mb-3">
                    <Clock className="w-6 h-6 text-blue-600" />
                    <h3 className="text-xl font-semibold text-blue-900 ml-3">Survival Prediction</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-2xl font-bold text-blue-900">
                        {prediction.survival_years}
                      </div>
                      <p className="text-blue-700">Years</p>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-blue-900">
                        {prediction.survival_months}
                      </div>
                      <p className="text-blue-700">Months</p>
                    </div>
                  </div>
                </div>

                {/* Confidence Score */}
                <div className="bg-gray-50 border-2 border-gray-200 p-6 rounded-xl">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Model Confidence</h3>
                  <div className="flex items-center">
                    <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                      <div
                        className="bg-indigo-600 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${prediction.confidence_score * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-lg font-semibold text-gray-900">
                      {Math.round(prediction.confidence_score * 100)}%
                    </span>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                  <div className="flex items-start">
                    <AlertCircle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-yellow-800">
                      <p className="font-medium mb-1">Medical Disclaimer</p>
                      <p>This prediction is for research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-gray-500">
          <p>&copy; 2025 AEGIS - AI-Enhanced Genomic Intelligence System</p>
        </div>
      </div>
    </div>
  );
};

export default AegisCancerPrediction;
# Subscription Conversion Predictor

## Overview
An advanced machine learning solution designed to optimize marketing campaigns in the banking sector through predictive analytics. This project transforms traditional mass marketing approaches into data-driven, targeted campaigns for Term Deposit subscriptions.

## Business Value
- Increased marketing ROI through precision targeting
- Reduced operational costs by identifying high-probability leads
- Enhanced customer engagement through personalized outreach
- Data-driven decision making for marketing strategies

## Technical Features

### Machine Learning Model
- **Algorithm**: Tuned XGBoost Classifier
- **Optimization**: F1-score optimization for balanced precision and recall
- **Class Imbalance**: Specialized handling of rare conversion cases
- **Pipeline**: Comprehensive preprocessing and feature engineering

### Interactive Web Interface
- **Framework**: Streamlit-based dashboard
- **Features**:
  - Real-time prediction capabilities
  - Scenario analysis tools
  - Adjustable decision threshold controls
  - Dynamic risk tolerance management

## Technical Requirements
- Python 3.8+
- Dependencies as specified in `requirements.txt`
- Adequate computational resources for model training

## Installation Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AryanAkhare/subscription-conversion-predictor.git
   cd subscription-conversion-predictor
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

1. **Model Generation**
   ```bash
   python deploy_model.py
   ```
   Note: Requires access to training dataset

2. **Launch Application**
   ```bash
   streamlit run app.py
   ```
   The web interface will automatically open in your default browser

## Model Performance
- Optimized for business-critical metrics
- Balanced precision and recall for practical application
- Validated through comprehensive cross-validation

## Future Enhancements
- Integration with real-time banking systems
- Advanced feature engineering capabilities
- Extended API functionality
- Enhanced visualization tools

## Contributing
Contributions are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## Author
**Aryan Akhare**

## License
This project is proprietary and confidential. All rights reserved.

---
For questions and support, please open an issue in the GitHub repository.
import pandas as pd


class RetentionAgent:
    """
    Local retention strategy generator.

    This version does NOT require a Groq API key.
    It generates a retention strategy using the customer's
    churn probability and available customer information.
    """

    def __init__(self):
        pass

    def generate_strategy(self, customer_data, churn_prob):
        """
        Generate a personalized retention strategy locally.
        """

        # Convert probability to percentage
        churn_percentage = round(churn_prob * 100, 2)

        # Safely read customer information
        tenure = customer_data.get("tenure", 0)
        monthly_charges = customer_data.get("MonthlyCharges", 0)
        total_charges = customer_data.get("TotalCharges", 0)

        contract = str(
            customer_data.get("Contract", "")
        ).lower()

        internet_service = str(
            customer_data.get("InternetService", "")
        ).lower()

        tech_support = str(
            customer_data.get("TechSupport", "")
        ).lower()

        payment_method = str(
            customer_data.get("PaymentMethod", "")
        ).lower()

        # Convert numeric values safely
        try:
            tenure = float(tenure)
        except:
            tenure = 0

        try:
            monthly_charges = float(monthly_charges)
        except:
            monthly_charges = 0

        try:
            total_charges = float(total_charges)
        except:
            total_charges = 0

        # --------------------------------------------------
        # Determine risk level
        # --------------------------------------------------

        if churn_prob >= 0.70:
            risk_level = "HIGH"
        elif churn_prob >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # --------------------------------------------------
        # Build retention recommendations
        # --------------------------------------------------

        actions = []

        # High churn risk
        if churn_prob >= 0.70:
            actions.append(
                "Prioritize the customer for proactive retention outreach."
            )

            actions.append(
                "Offer a personalized loyalty incentive or suitable plan adjustment."
            )

        elif churn_prob >= 0.40:
            actions.append(
                "Schedule a proactive customer-success follow-up."
            )

            actions.append(
                "Review the customer's plan and identify possible cost-saving options."
            )

        else:
            actions.append(
                "Continue regular customer engagement and monitor churn indicators."
            )

        # Month-to-month contract
        if "month-to-month" in contract:
            actions.append(
                "Consider offering a discounted 12-month contract to increase customer commitment."
            )

        # Fiber customer
        if "fiber" in internet_service:
            actions.append(
                "Check internet performance and technical-support history; consider a technical health check."
            )

        # No technical support
        if tech_support in ["no", ""]:
            actions.append(
                "Offer technical support assistance and verify whether the current service meets expectations."
            )

        # High monthly charges
        if monthly_charges >= 80:
            actions.append(
                "Review the current plan for possible right-sizing or a loyalty discount."
            )

        # New customer
        if tenure <= 6:
            actions.append(
                "Provide onboarding support and check for early-service dissatisfaction."
            )

        # Long-term customer
        if tenure >= 36:
            actions.append(
                "Recognize long-term loyalty and consider a personalized loyalty benefit."
            )

        # Electronic/check payment related
        if "electronic check" in payment_method:
            actions.append(
                "Review payment experience and provide easier automatic-payment options if appropriate."
            )

        # --------------------------------------------------
        # Final strategy
        # --------------------------------------------------

        strategy = f"""
RETENTION STRATEGY
==================

Customer Churn Probability: {churn_percentage}%
Risk Level: {risk_level}

CUSTOMER PROFILE
----------------
Tenure: {tenure:.0f} months
Monthly Charges: {monthly_charges:.2f}
Total Charges: {total_charges:.2f}
Contract: {contract if contract else "Not available"}
Internet Service: {internet_service if internet_service else "Not available"}

RECOMMENDED ACTIONS
-------------------
"""

        for i, action in enumerate(actions, 1):
            strategy += f"{i}. {action}\n"

        strategy += f"""
RETENTION OBJECTIVE
-------------------
Reduce the customer's likelihood of churn by addressing
the customer's service, pricing, contract, and engagement
risk factors.

PRIORITY
--------
{risk_level} RISK CUSTOMER
"""

        return strategy.strip()


def run_retention_flow(customer_id, df, model):
    """
    Runs the local legacy retention flow.

    No Groq API key is required.
    """

    # Use the first customer row for the existing demo workflow
    customer_data = df.iloc[0].to_dict()

    # Remove target column before prediction
    prediction_data = df.drop(
        columns=["Churn"],
        errors="ignore"
    )

    # Calculate churn probability
    prob = model.predict_proba(
        prediction_data
    )[0][1]

    # Create local retention agent
    agent = RetentionAgent()

    # Generate strategy
    return agent.generate_strategy(
        customer_data,
        prob
    )
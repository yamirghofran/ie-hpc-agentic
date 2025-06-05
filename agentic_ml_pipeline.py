"""
Agentic ML Pipeline using LangGraph for Titanic Survival Classification
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple # Added Callable, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"agentic_ml_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class MLState(TypedDict):
    """State for the ML pipeline"""
    data: Optional[pd.DataFrame]
    X_train: Optional[pd.DataFrame]
    X_val: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_val: Optional[pd.Series]
    y_test: Optional[pd.Series]
    model: Optional[Any]
    metrics: Dict[str, Any]
    features: List[str]
    logs: List[str]
    current_step: str
    error: Optional[str]
    config: Dict[str, Any]  # New
    iteration_count: int     # New
    tool_calls: List[Dict[str, Any]] # New
    agent_scratchpad: List[BaseMessage] # New for LLM messages


@dataclass
class Tool:
    name: str
    description: str
    func: Callable

class ToolExecutor:
    def __init__(self, pipeline_instance: 'AgenticMLPipeline'):
        self.tools: Dict[str, Tool] = {}
        self.pipeline = pipeline_instance # To access logger, _log_action etc.

    def register_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = ""):
        """Registers a function as a tool."""
        tool_name = name if name else func.__name__
        tool_description = description if description else func.__doc__ if func.__doc__ else "No description provided."
        
        if tool_name in self.tools:
            raise ValueError(f"Tool with name '{tool_name}' already registered.")
        
        self.tools[tool_name] = Tool(name=tool_name, description=tool_description, func=func)
        self.pipeline.logger.info(f"Tool '{tool_name}' registered.")
        # No need to return func as it's not used as a decorator in this context anymore

    def execute_tool(self, tool_name: str, state: MLState, **kwargs) -> MLState:
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found."
            self.pipeline.logger.error(error_msg)
            state["error"] = error_msg
            state["tool_calls"].append({
                "tool_name": tool_name,
                "arguments": kwargs,
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            return self.pipeline._log_action("ToolExecutor", f"Execute {tool_name}", f"Error: {error_msg}", state)

        tool = self.tools[tool_name]
        self.pipeline.logger.info(f"Executing tool: {tool.name} with args: {kwargs}")
        
        tool_call_log = {
            "tool_name": tool.name,
            "arguments": kwargs,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        state["tool_calls"].append(tool_call_log)
        
        try:
            # Tools are methods of AgenticMLPipeline, their 'self' is already bound.
            updated_state = tool.func(state, **kwargs)
            
            outcome_message = f"Tool '{tool.name}' executed successfully."
            tool_call_log["status"] = "success"
            # More detailed result can be added by the tool itself to the state if needed
            tool_call_log["result_summary"] = f"Tool {tool.name} completed."
            
            self.pipeline.logger.info(outcome_message)
            return updated_state

        except Exception as e:
            error_msg = f"Error executing tool '{tool.name}': {str(e)}"
            self.pipeline.logger.error(error_msg, exc_info=True)
            updated_state = state.copy()
            updated_state["error"] = error_msg
            
            tool_call_log["status"] = "error"
            tool_call_log["error"] = str(e)
            
            return self.pipeline._log_action("ToolExecutor", f"Execute {tool.name}", f"Error: {str(e)}", updated_state)

class AgenticMLPipeline:
    """Agentic ML Pipeline using LangGraph"""
    
    def __init__(self, data_path: str = "titanic3.xls", seed: int = 42, target_auc: float = 0.80, max_iterations: int = 3): # Added target_auc, max_iterations
        self.data_path = data_path
        self.seed = seed
        self.logger = setup_logging()
        self.config = { # New
            "target_auc": target_auc,
            "max_iterations": max_iterations
        }
        self.tool_executor = ToolExecutor(pipeline_instance=self) # New
        # Initialize LLM (ensure OPENAI_API_KEY is set in environment)
        try:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
            self.logger.info("ChatOpenAI (gpt-4o) initialized for LLM tasks.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatOpenAI: {e}. LLM-driven agents may not work.")
            self.llm = None # Allow pipeline to run without LLM if init fails, agents should handle self.llm being None.

        self._register_internal_tools() # New: Call a method to register tools
        self.graph = self._build_graph()

    def _register_internal_tools(self):
        """Registers all tool methods with the ToolExecutor."""
        self.tool_executor.register_tool(
            func=self._tool_load_data,
            name="load_data",
            description="Loads data from an Excel file."
        )
        self.tool_executor.register_tool(
            func=self._tool_initial_exploration,
            name="initial_exploration",
            description="Performs initial data exploration (shape, missing values)."
        )
        self.tool_executor.register_tool(
            func=self._tool_analyze_numeric_features,
            name="analyze_numeric_features",
            description="Identifies and logs numeric columns in the dataset."
        )
        self.tool_executor.register_tool(
            func=self._tool_detect_fare_outliers,
            name="detect_fare_outliers",
            description="Detects and logs outliers in the 'fare' column using the IQR method."
        )
        self.tool_executor.register_tool(
            func=self._tool_analyze_target_distribution,
            name="analyze_target_distribution",
            description="Analyzes and logs the distribution of the 'survived' target variable."
        )
        self.tool_executor.register_tool(
            func=self._tool_drop_features,
            name="drop_features",
            description="Drops specified columns from the dataset."
        )
        self.tool_executor.register_tool(
            func=self._tool_impute_home_dest_with_ticket,
            name="impute_home_dest_with_ticket",
            description="Imputes 'home.dest' using 'ticket' information and fills remaining NaNs with 'Unknown'."
        )
        self.tool_executor.register_tool(
            func=self._tool_extract_title_from_name,
            name="extract_title_from_name",
            description="Extracts 'title' from 'name' column and categorizes rare titles."
        )
        self.tool_executor.register_tool(
            func=self._tool_create_ticket_count_feature,
            name="create_ticket_count_feature",
            description="Creates a 'ticket_count' feature based on ticket frequency."
        )
        self.tool_executor.register_tool(
            func=self._tool_split_data,
            name="split_data",
            description="Splits data into training, validation, and test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_impute_age_by_group,
            name="impute_age_by_group",
            description="Imputes 'age' using median by 'pclass' and 'sex' groups for train, val, test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_impute_fare_median,
            name="impute_fare_median",
            description="Imputes 'fare' using its median for train, val, test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_impute_embarked_mode,
            name="impute_embarked_mode",
            description="Imputes 'embarked' with mode 'S' for train, val, test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_extract_deck_from_cabin,
            name="extract_deck_from_cabin",
            description="Extracts 'deck' from 'cabin' and drops 'cabin' for train, val, test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_impute_deck_knn,
            name="impute_deck_knn",
            description="Imputes 'deck' using KNN based on 'pclass', 'fare', 'age' for train, val, test sets."
        )
        self.tool_executor.register_tool(
            func=self._tool_group_rare_home_dest,
            name="group_rare_home_dest",
            description="Groups rare 'home.dest' values into 'Other'."
        )
        self.tool_executor.register_tool(
            func=self._tool_one_hot_encode_features,
            name="one_hot_encode_features",
            description="One-hot encodes specified categorical features."
        )
        self.tool_executor.register_tool(
            func=self._tool_scale_features,
            name="scale_features",
            description="Scales specified features using StandardScaler or MinMaxScaler."
        )
        self.tool_executor.register_tool(
            func=self._tool_balance_data_adasyn,
            name="balance_data_adasyn",
            description="Balances X_train and y_train using ADASYN."
        )
        self.tool_executor.register_tool(
            func=self._tool_variance_threshold_select,
            name="variance_threshold_select",
            description="Selects features using VarianceThreshold and updates datasets."
        )
        self.tool_executor.register_tool(
            func=self._tool_mutual_information_select,
            name="mutual_information_select",
            description="Selects top N features based on mutual information and updates datasets."
        )
        self.tool_executor.register_tool(
            func=self._tool_train_model,
            name="train_model",
            description="Trains a specified model (e.g., LogisticRegression) with given parameters."
        )
        self.tool_executor.register_tool(
            func=self._tool_evaluate_model,
            name="evaluate_model",
            description="Evaluates the trained model on validation and test sets, calculating accuracy and AUC."
        )
        # ... other tools will be registered here later
        
    # --- Tool Implementations ---
    def _tool_load_data(self, state: MLState, data_path: str) -> MLState:
        """Loads data from the specified path (Excel file)."""
        tool_name = "Tool:load_data"
        try:
            self.logger.info(f"{tool_name}: Attempting to load data from {data_path}")
            data = pd.read_excel(data_path)
            state["data"] = data
            outcome = f"Successfully loaded {len(data)} records with {len(data.columns)} columns from {data_path}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Load Excel File", outcome, state)
        except Exception as e:
            error_msg = f"Error loading data from {data_path}: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Load Excel File", f"Error: {str(e)}", state)
        return state

    def _tool_initial_exploration(self, state: MLState) -> MLState:
        """Performs initial data exploration (shape, missing values)."""
        tool_name = "Tool:initial_exploration"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            # Ensure 'data' key exists if it was expected but not found due to prior error
            if state.get("data") is None and not state.get("error"):
                 state["error"] = f"{tool_name}: Data not found in state for exploration."
            return state
        
        try:
            data = state["data"]
            self.logger.info(f"{tool_name}: Performing initial exploration on data with shape {data.shape}")
            missing_info = data.isnull().sum()
            missing_cols_summary = missing_info[missing_info > 0].to_dict()
            outcome = f"Data shape: {data.shape}, Missing values found in: {missing_cols_summary if missing_cols_summary else 'None'}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Explore Data (Shape, Missing)", outcome, state)
        except Exception as e:
            error_msg = f"Error during initial data exploration: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Explore Data (Shape, Missing)", f"Error: {str(e)}", state)
        return state

    # --- End of Tool Implementations ---

    # --- EDA Tool Implementations ---
    def _tool_analyze_numeric_features(self, state: MLState) -> MLState:
        tool_name = "Tool:analyze_numeric_features"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"]
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outcome = f"Numeric columns identified: {list(numeric_cols)}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Analyze Numeric Features", outcome, state)
        except Exception as e:
            error_msg = f"Error analyzing numeric features: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Analyze Numeric Features", f"Error: {str(e)}", state)
        return state

    def _tool_detect_fare_outliers(self, state: MLState) -> MLState:
        tool_name = "Tool:detect_fare_outliers"
        if state.get("error") or state.get("data") is None or 'fare' not in state["data"].columns:
            self.logger.warning(f"{tool_name}: Skipping due to previous error, no data, or 'fare' column missing.")
            if 'fare' not in state.get("data", pd.DataFrame()).columns and not state.get("error"): # check if data exists before checking columns
                 state["error"] = f"{tool_name}: 'fare' column not found in data."
            return state
        try:
            data = state["data"]
            Q1 = data['fare'].quantile(0.25)
            Q3 = data['fare'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data['fare'] < Q1 - 1.5*IQR) | (data['fare'] > Q3 + 1.5*IQR)]
            outcome = f"Found {len(outliers)} outliers in 'fare' column (IQR method)."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Detect Fare Outliers", outcome, state)
        except Exception as e:
            error_msg = f"Error detecting fare outliers: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Detect Fare Outliers", f"Error: {str(e)}", state)
        return state

    def _tool_analyze_target_distribution(self, state: MLState) -> MLState:
        tool_name = "Tool:analyze_target_distribution"
        if state.get("error") or state.get("data") is None or 'survived' not in state["data"].columns:
            self.logger.warning(f"{tool_name}: Skipping due to previous error, no data, or 'survived' column missing.")
            if 'survived' not in state.get("data", pd.DataFrame()).columns and not state.get("error"):
                 state["error"] = f"{tool_name}: 'survived' column not found in data."
            return state
        try:
            data = state["data"]
            target_dist = data['survived'].value_counts()
            outcome = f"Target 'survived' distribution - Died: {target_dist.get(0,0)}, Survived: {target_dist.get(1,0)}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Analyze Target Distribution", outcome, state)
        except Exception as e:
            error_msg = f"Error analyzing target distribution: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Analyze Target Distribution", f"Error: {str(e)}", state)
        return state
    # --- End of EDA Tool Implementations ---

    # --- Feature Engineering Tool Implementations ---
    def _tool_drop_features(self, state: MLState, columns: List[str], agent_name_context: str = "FeatureEngineering") -> MLState:
        tool_name = f"Tool:drop_features (context: {agent_name_context})"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"].copy() # Operate on a copy
            original_columns = set(data.columns)
            data.drop(columns=columns, axis=1, inplace=True, errors='ignore')
            dropped_cols = list(original_columns - set(data.columns))
            state["data"] = data
            outcome = f"Dropped columns: {dropped_cols}. Current columns: {list(data.columns)}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, f"Drop Columns: {columns}", outcome, state)
        except Exception as e:
            error_msg = f"Error dropping columns {columns}: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, f"Drop Columns: {columns}", f"Error: {str(e)}", state)
        return state

    def _tool_impute_home_dest_with_ticket(self, state: MLState) -> MLState:
        tool_name = "Tool:impute_home_dest_with_ticket"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"].copy()
            if 'home.dest' not in data.columns or 'ticket' not in data.columns:
                outcome = "Skipped: 'home.dest' or 'ticket' column not found."
                self.logger.warning(f"{tool_name}: {outcome}")
                state = self._log_action(tool_name, "Impute home.dest", outcome, state)
                if not state.get("error"): # only set error if not already set
                     state["error"] = outcome
                return state

            ticket_to_dest = data[data['home.dest'].notna()].groupby('ticket')['home.dest'].first()
            data.loc[data['home.dest'].isnull(), 'home.dest'] = data.loc[data['home.dest'].isnull(), 'ticket'].map(ticket_to_dest)
            data['home.dest'] = data['home.dest'].fillna('Unknown')
            state["data"] = data
            outcome = f"Imputed 'home.dest' using 'ticket' mapping. Missing values remaining: {data['home.dest'].isnull().sum()}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Impute home.dest", outcome, state)
        except Exception as e:
            error_msg = f"Error imputing 'home.dest': {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Impute home.dest", f"Error: {str(e)}", state)
        return state

    def _tool_extract_title_from_name(self, state: MLState) -> MLState:
        tool_name = "Tool:extract_title_from_name"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"].copy()
            if 'name' not in data.columns:
                outcome = "Skipped: 'name' column not found."
                self.logger.warning(f"{tool_name}: {outcome}")
                state = self._log_action(tool_name, "Extract Title", outcome, state)
                if not state.get("error"):
                    state["error"] = outcome
                return state

            data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
            data['title'] = data['title'].apply(lambda x: x if x in common_titles else 'Rare')
            data['title'] = data['title'].fillna('Unknown') # Handle cases where regex might not match
            state["data"] = data
            outcome = f"Extracted 'title' feature. Categories: {list(data['title'].unique())}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Extract Title from Name", outcome, state)
        except Exception as e:
            error_msg = f"Error extracting title from name: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Extract Title from Name", f"Error: {str(e)}", state)
        return state

    def _tool_create_ticket_count_feature(self, state: MLState) -> MLState:
        tool_name = "Tool:create_ticket_count_feature"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"].copy()
            if 'ticket' not in data.columns:
                outcome = "Skipped: 'ticket' column not found."
                self.logger.warning(f"{tool_name}: {outcome}")
                state = self._log_action(tool_name, "Create Ticket Count", outcome, state)
                if not state.get("error"):
                    state["error"] = outcome
                return state
                
            data['ticket_count'] = data['ticket'].map(data['ticket'].value_counts()).fillna(1) # fillna for safety
            state["data"] = data
            outcome = f"Created 'ticket_count' feature. Range: {data['ticket_count'].min()}-{data['ticket_count'].max()}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Create Ticket Count Feature", outcome, state)
        except Exception as e:
            error_msg = f"Error creating 'ticket_count' feature: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Create Ticket Count Feature", f"Error: {str(e)}", state)
        return state
    # --- End of Feature Engineering Tool Implementations ---

    # --- Data Imputation Tool Implementations ---
    def _tool_split_data(self, state: MLState) -> MLState:
        tool_name = "Tool:split_data"
        if state.get("error") or state.get("data") is None:
            self.logger.warning(f"{tool_name}: Skipping due to previous error or no data.")
            return state
        try:
            data = state["data"].copy()
            X = data.drop('survived', axis=1)
            y = data['survived']
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=self.seed
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.seed
            )
            state["X_train"], state["X_val"], state["X_test"] = X_train, X_val, X_test
            state["y_train"], state["y_val"], state["y_test"] = y_train, y_val, y_test
            outcome = f"Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Split Dataset", outcome, state)
        except Exception as e:
            error_msg = f"Error splitting data: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Split Dataset", f"Error: {str(e)}", state)
        return state

    def _tool_impute_age_by_group(self, state: MLState) -> MLState:
        tool_name = "Tool:impute_age_by_group"
        datasets = {"X_train": state.get("X_train"), "X_val": state.get("X_val"), "X_test": state.get("X_test")}
        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in datasets.values()):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: One or more datasets (X_train, X_val, X_test) are missing or invalid."
            return state
        try:
            for name, df_orig in datasets.items():
                if df_orig is None: continue
                df = df_orig.copy()
                if 'age' in df.columns and 'pclass' in df.columns and 'sex' in df.columns:
                    df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
                    # Handle cases where a group median might still be NaN (e.g., all NaNs in a group)
                    if df['age'].isnull().any():
                        df['age'].fillna(df['age'].median(), inplace=True) # Fallback to global median
                    state[name] = df
                else:
                    self.logger.warning(f"{tool_name}: 'age', 'pclass', or 'sex' missing in {name}, skipping age imputation for it.")

            outcome = "Imputed 'age' using median by 'pclass' and 'sex' (with global median fallback)."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Impute Age by Group", outcome, state)
        except Exception as e:
            error_msg = f"Error imputing age by group: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Impute Age by Group", f"Error: {str(e)}", state)
        return state

    def _tool_impute_fare_median(self, state: MLState) -> MLState:
        tool_name = "Tool:impute_fare_median"
        datasets = {"X_train": state.get("X_train"), "X_val": state.get("X_val"), "X_test": state.get("X_test")}
        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in datasets.values()):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: One or more datasets (X_train, X_val, X_test) are missing or invalid."
            return state
        try:
            for name, df_orig in datasets.items():
                if df_orig is None: continue
                df = df_orig.copy()
                if 'fare' in df.columns and df['fare'].isnull().any():
                    df['fare'].fillna(df['fare'].median(), inplace=True)
                    state[name] = df
            outcome = "Imputed 'fare' using median where missing."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Impute Fare Median", outcome, state)
        except Exception as e:
            error_msg = f"Error imputing fare with median: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Impute Fare Median", f"Error: {str(e)}", state)
        return state

    def _tool_impute_embarked_mode(self, state: MLState) -> MLState:
        tool_name = "Tool:impute_embarked_mode"
        datasets = {"X_train": state.get("X_train"), "X_val": state.get("X_val"), "X_test": state.get("X_test")}
        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in datasets.values()):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: One or more datasets (X_train, X_val, X_test) are missing or invalid."
            return state
        try:
            for name, df_orig in datasets.items():
                if df_orig is None: continue
                df = df_orig.copy()
                if 'embarked' in df.columns:
                    df['embarked'].fillna('S', inplace=True) # 'S' is the mode
                    state[name] = df
            outcome = "Imputed 'embarked' with mode 'S' where missing."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Impute Embarked Mode", outcome, state)
        except Exception as e:
            error_msg = f"Error imputing embarked with mode: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Impute Embarked Mode", f"Error: {str(e)}", state)
        return state

    def _tool_extract_deck_from_cabin(self, state: MLState) -> MLState:
        tool_name = "Tool:extract_deck_from_cabin"
        datasets = {"X_train": state.get("X_train"), "X_val": state.get("X_val"), "X_test": state.get("X_test")}
        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in datasets.values()):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: One or more datasets (X_train, X_val, X_test) are missing or invalid."
            return state
        try:
            for name, df_orig in datasets.items():
                if df_orig is None: continue
                df = df_orig.copy()
                if 'cabin' in df.columns:
                    df['deck'] = df['cabin'].str[0]
                    df.drop('cabin', axis=1, inplace=True)
                else: # Ensure 'deck' column exists even if 'cabin' didn't
                    df['deck'] = None
                state[name] = df
            outcome = "Extracted 'deck' from 'cabin' (if exists) and dropped 'cabin'."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Extract Deck from Cabin", outcome, state)
        except Exception as e:
            error_msg = f"Error extracting deck from cabin: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Extract Deck from Cabin", f"Error: {str(e)}", state)
        return state

    def _tool_impute_deck_knn(self, state: MLState) -> MLState:
        tool_name = "Tool:impute_deck_knn"
        X_train_orig, X_val_orig, X_test_orig = state.get("X_train"), state.get("X_val"), state.get("X_test")

        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in [X_train_orig, X_val_orig, X_test_orig]):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train, X_val, or X_test is missing or invalid for KNN deck imputation."
            return state
        
        X_train, X_val, X_test = X_train_orig.copy(), X_val_orig.copy(), X_test_orig.copy()

        try:
            clustering_features = ['pclass', 'fare', 'age']
            if not all(feat in X_train.columns for feat in clustering_features):
                missing_req_cols = [feat for feat in clustering_features if feat not in X_train.columns]
                outcome = f"Skipped KNN deck imputation: Required clustering features missing: {missing_req_cols}."
                self.logger.warning(f"{tool_name}: {outcome}")
                # Fill with 'Unknown' as a fallback if KNN cannot be performed
                for df_name, df_obj in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
                    if 'deck' in df_obj.columns: df_obj['deck'].fillna('Unknown', inplace=True)
                    state[df_name] = df_obj
                state = self._log_action(tool_name, "Impute Deck KNN", outcome, state)
                if not state.get("error"): state["error"] = outcome # Set error if not already set
                return state

            # Ensure no NaNs in clustering features for KNN (use median imputation as a quick fix)
            for df_knn in [X_train, X_val, X_test]:
                for col in clustering_features:
                    if df_knn[col].isnull().any():
                        df_knn[col].fillna(df_knn[col].median(), inplace=True)
            
            scaler = StandardScaler()
            X_train_scaled_cluster_feats = scaler.fit_transform(X_train[clustering_features])
            
            known_deck_mask = X_train['deck'].notna() & (X_train['deck'] != 'Unknown') # Ensure 'Unknown' is not treated as known
            
            if known_deck_mask.sum() > 1: # Need at least 2 samples to fit KNN meaningfully
                knn = KNeighborsClassifier(n_neighbors=min(5, known_deck_mask.sum())) # Adjust n_neighbors if few knowns
                knn.fit(X_train_scaled_cluster_feats[known_deck_mask], X_train.loc[known_deck_mask, 'deck'])
                
                for df_name, df_obj in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
                    scaled_data = scaler.transform(df_obj[clustering_features])
                    unknown_mask = df_obj['deck'].isna() | (df_obj['deck'] == 'Unknown')
                    if unknown_mask.sum() > 0:
                        predicted_decks = knn.predict(scaled_data[unknown_mask])
                        df_obj.loc[unknown_mask, 'deck'] = predicted_decks
                    df_obj['deck'].fillna('Unknown', inplace=True) # Fill any remaining NaNs (e.g. if all were unknown initially)
                    state[df_name] = df_obj
                outcome = "Imputed 'deck' using KNN (or 'Unknown' if insufficient data)."
            else:
                for df_name, df_obj in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
                    if 'deck' in df_obj.columns: df_obj['deck'].fillna('Unknown', inplace=True)
                    state[df_name] = df_obj
                outcome = "Insufficient known 'deck' values for KNN imputation; filled with 'Unknown'."
            
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Impute Deck KNN", outcome, state)

        except Exception as e:
            error_msg = f"Error imputing deck with KNN: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            # Fallback: fill deck with 'Unknown' on error
            for df_name, df_obj in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
                if df_obj is not None and 'deck' in df_obj.columns: df_obj['deck'].fillna('Unknown', inplace=True)
                state[df_name] = df_obj
            state = self._log_action(tool_name, "Impute Deck KNN", f"Error: {str(e)}, filled with 'Unknown'", state)
        return state
    # --- End of Data Imputation Tool Implementations ---

    # --- Preprocessing Tool Implementations ---
    def _tool_group_rare_home_dest(self, state: MLState) -> MLState:
        tool_name = "Tool:group_rare_home_dest"
        datasets_keys = ["X_train", "X_val", "X_test"]
        if state.get("error") or not all(isinstance(state.get(key), pd.DataFrame) for key in datasets_keys):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: One or more datasets (X_train, X_val, X_test) are missing."
            return state
        try:
            major_destinations = ['New York, NY', 'London', 'Paris, France', 'Montreal, PQ', 'Unknown']
            for key in datasets_keys:
                df = state[key].copy()
                if 'home.dest' in df.columns:
                    df['home.dest'] = df['home.dest'].apply(lambda x: x if x in major_destinations else 'Other')
                    state[key] = df
                else:
                    self.logger.warning(f"{tool_name}: 'home.dest' column not found in {key}. Skipping for this dataset.")
            
            outcome = "Grouped rare 'home.dest' values into 'Other'."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Group Rare home.dest", outcome, state)
        except Exception as e:
            error_msg = f"Error grouping rare 'home.dest' values: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Group Rare home.dest", f"Error: {str(e)}", state)
        return state

    def _tool_one_hot_encode_features(self, state: MLState, categorical_cols: List[str]) -> MLState:
        tool_name = "Tool:one_hot_encode_features"
        X_train_orig, X_val_orig, X_test_orig = state.get("X_train"), state.get("X_val"), state.get("X_test")

        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in [X_train_orig, X_val_orig, X_test_orig]):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train, X_val, or X_test is missing for OHE."
            return state
        
        X_train, X_val, X_test = X_train_orig.copy(), X_val_orig.copy(), X_test_orig.copy()
        
        actual_categorical_cols = [col for col in categorical_cols if col in X_train.columns]
        if not actual_categorical_cols:
            self.logger.warning(f"{tool_name}: No specified categorical columns found in X_train. Skipping OHE.")
            state = self._log_action(tool_name, "One-Hot Encode", "Skipped - no relevant columns", state)
            return state

        try:
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            
            encoded_train = encoder.fit_transform(X_train[actual_categorical_cols])
            
            feature_names = []
            for i, col in enumerate(actual_categorical_cols):
                # Handle cases where a category might be dropped by 'drop=first' or not present after fit
                cats = encoder.categories_[i]
                # If drop='first', the first category is dropped, so feature names correspond to cats[1:]
                # If a category was all NaNs and then imputed, it might appear.
                # handle_unknown='ignore' means new categories in val/test become all zeros for these features.
                valid_cats = cats[1:] if encoder.drop == 'first' else cats
                if hasattr(encoder, 'feature_names_in_') and hasattr(encoder, 'get_feature_names_out'):
                     # Modern scikit-learn
                    try:
                        current_col_feature_names = encoder.get_feature_names_out([col])
                        # Adjust if 'drop=first' was used, get_feature_names_out handles this.
                        # We need to ensure we only take names for the current column if encoder was fit on multiple.
                        # This part is tricky if encoder is fit on all actual_categorical_cols at once.
                        # Let's get all feature names and then filter.
                    except Exception: # Fallback for older or different encoder behavior
                        current_col_feature_names = [f"{col}_{cat_val}" for cat_val in valid_cats]
                else: # Older scikit-learn or manual construction
                    current_col_feature_names = [f"{col}_{cat_val}" for cat_val in valid_cats]

                # This logic for feature_names needs to be robust for multi-column fit
            # A simpler way for multi-column fit:
            if hasattr(encoder, 'get_feature_names_out'):
                feature_names = list(encoder.get_feature_names_out(actual_categorical_cols))
            else: # Manual construction if get_feature_names_out is not available
                feature_names = []
                temp_encoder_for_names = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                temp_encoder_for_names.fit(X_train[actual_categorical_cols]) # Fit again just for names, not ideal
                for i, col in enumerate(actual_categorical_cols):
                    cats = temp_encoder_for_names.categories_[i][1:] # Assuming drop='first'
                    feature_names.extend([f"{col}_{cat}" for cat in cats])


            encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=X_train.index)
            X_train = pd.concat([X_train.drop(actual_categorical_cols, axis=1), encoded_train_df], axis=1)
            
            if X_val is not None:
                encoded_val = encoder.transform(X_val[actual_categorical_cols])
                encoded_val_df = pd.DataFrame(encoded_val, columns=feature_names, index=X_val.index)
                X_val = pd.concat([X_val.drop(actual_categorical_cols, axis=1), encoded_val_df], axis=1)
            
            if X_test is not None:
                encoded_test = encoder.transform(X_test[actual_categorical_cols])
                encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=X_test.index)
                X_test = pd.concat([X_test.drop(actual_categorical_cols, axis=1), encoded_test_df], axis=1)

            state["X_train"], state["X_val"], state["X_test"] = X_train, X_val, X_test
            outcome = f"One-hot encoded {len(actual_categorical_cols)} features, resulting in {len(feature_names)} new OHE features."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "One-Hot Encode Features", outcome, state)
        except Exception as e:
            error_msg = f"Error one-hot encoding features {actual_categorical_cols}: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "One-Hot Encode Features", f"Error: {str(e)}", state)
        return state

    def _tool_scale_features(self, state: MLState, cols_to_scale: List[str], scaler_type: str) -> MLState:
        tool_name = f"Tool:scale_features ({scaler_type})"
        X_train_orig, X_val_orig, X_test_orig = state.get("X_train"), state.get("X_val"), state.get("X_test")

        if state.get("error") or not all(isinstance(df, pd.DataFrame) for df in [X_train_orig, X_val_orig, X_test_orig]):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid datasets.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train, X_val, or X_test is missing for scaling."
            return state

        X_train, X_val, X_test = X_train_orig.copy(), X_val_orig.copy(), X_test_orig.copy()
        
        actual_cols_to_scale_train = [col for col in cols_to_scale if col in X_train.columns]
        if not actual_cols_to_scale_train:
            self.logger.warning(f"{tool_name}: No specified columns for scaling found in X_train. Skipping.")
            state = self._log_action(tool_name, f"Scale Features ({scaler_type})", "Skipped - no relevant columns in X_train", state)
            return state
            
        try:
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaler_type: {scaler_type}")

            X_train[actual_cols_to_scale_train] = scaler.fit_transform(X_train[actual_cols_to_scale_train])
            
            if X_val is not None:
                actual_cols_to_scale_val = [col for col in cols_to_scale if col in X_val.columns]
                if actual_cols_to_scale_val:
                     X_val[actual_cols_to_scale_val] = scaler.transform(X_val[actual_cols_to_scale_val])
            
            if X_test is not None:
                actual_cols_to_scale_test = [col for col in cols_to_scale if col in X_test.columns]
                if actual_cols_to_scale_test:
                    X_test[actual_cols_to_scale_test] = scaler.transform(X_test[actual_cols_to_scale_test])
            
            state["X_train"], state["X_val"], state["X_test"] = X_train, X_val, X_test
            outcome = f"Scaled {len(actual_cols_to_scale_train)} features using {scaler_type}."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, f"Scale Features ({scaler_type})", outcome, state)
        except Exception as e:
            error_msg = f"Error scaling features {actual_cols_to_scale_train} with {scaler_type}: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, f"Scale Features ({scaler_type})", f"Error: {str(e)}", state)
        return state
    # --- End of Preprocessing Tool Implementations ---

    # --- Feature Selection Tool Implementations ---
    def _tool_balance_data_adasyn(self, state: MLState) -> MLState:
        tool_name = "Tool:balance_data_adasyn"
        X_train_orig, y_train_orig = state.get("X_train"), state.get("y_train")

        if state.get("error") or not isinstance(X_train_orig, pd.DataFrame) or not isinstance(y_train_orig, pd.Series):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid X_train/y_train.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train or y_train is missing/invalid for ADASYN."
            return state
        
        X_train, y_train = X_train_orig.copy(), y_train_orig.copy()

        try:
            # Ensure no NaNs in X_train before ADASYN, as it can cause issues.
            # A simple strategy: fill with median for numeric columns.
            # This should ideally be handled more robustly or ensured by prior steps.
            if X_train.isnull().sum().sum() > 0:
                self.logger.warning(f"{tool_name}: X_train contains NaNs before ADASYN. Attempting median fill for numeric columns.")
                for col in X_train.select_dtypes(include=np.number).columns:
                    if X_train[col].isnull().any():
                        X_train[col].fillna(X_train[col].median(), inplace=True)
                if X_train.isnull().sum().sum() > 0: # Check again
                    raise ValueError("NaNs remain in X_train after attempting median fill; ADASYN cannot proceed.")

            adasyn = ADASYN(random_state=self.seed)
            X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
            
            state["X_train"] = pd.DataFrame(X_train_balanced, columns=X_train.columns) # Ensure DataFrame
            state["y_train"] = pd.Series(y_train_balanced, name=y_train.name) # Ensure Series
            
            outcome = f"Balanced training data using ADASYN. Original size: {len(X_train)}, New size: {len(X_train_balanced)}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Balance Data (ADASYN)", outcome, state)
        except Exception as e:
            error_msg = f"Error balancing data with ADASYN: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            # Revert to original X_train, y_train if balancing failed to avoid downstream issues with potentially modified but failed state
            state["X_train"] = X_train_orig
            state["y_train"] = y_train_orig
            state = self._log_action(tool_name, "Balance Data (ADASYN)", f"Error: {str(e)}", state)
        return state

    def _tool_variance_threshold_select(self, state: MLState, threshold: float = 0.01) -> MLState:
        tool_name = "Tool:variance_threshold_select"
        X_train_orig, X_val_orig, X_test_orig = state.get("X_train"), state.get("X_val"), state.get("X_test")

        if state.get("error") or not isinstance(X_train_orig, pd.DataFrame):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid X_train.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train is missing/invalid for VarianceThreshold."
            return state
        
        X_train = X_train_orig.copy()
        X_val = X_val_orig.copy() if isinstance(X_val_orig, pd.DataFrame) else None
        X_test = X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else None

        try:
            # Ensure no NaNs before VarianceThreshold
            if X_train.isnull().sum().sum() > 0:
                self.logger.warning(f"{tool_name}: X_train contains NaNs before VarianceThreshold. Attempting median fill for numeric columns.")
                for col in X_train.select_dtypes(include=np.number).columns:
                    if X_train[col].isnull().any(): X_train[col].fillna(X_train[col].median(), inplace=True)
                if X_train.isnull().sum().sum() > 0: raise ValueError("NaNs remain in X_train for VarianceThreshold.")

            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X_train)
            
            selected_features_mask = selector.get_support()
            selected_features = X_train.columns[selected_features_mask].tolist()
            dropped_features = X_train.columns[~selected_features_mask].tolist()

            state["X_train"] = X_train[selected_features]
            if X_val is not None: state["X_val"] = X_val[selected_features]
            if X_test is not None: state["X_test"] = X_test[selected_features]
            state["features"] = selected_features # Update overall selected features list
            
            outcome = f"Applied VarianceThreshold(threshold={threshold}). Selected {len(selected_features)} features. Dropped: {dropped_features if dropped_features else 'None'}."
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Variance Threshold Selection", outcome, state)
        except Exception as e:
            error_msg = f"Error during VarianceThreshold selection: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            # Revert to original datasets if failed
            state["X_train"], state["X_val"], state["X_test"] = X_train_orig, X_val_orig, X_test_orig
            state["features"] = list(X_train_orig.columns) # Revert features list
            state = self._log_action(tool_name, "Variance Threshold Selection", f"Error: {str(e)}", state)
        return state

    def _tool_mutual_information_select(self, state: MLState, num_top_features: int = 15) -> MLState:
        tool_name = "Tool:mutual_information_select"
        X_train_orig, y_train_orig = state.get("X_train"), state.get("y_train")
        X_val_orig, X_test_orig = state.get("X_val"), state.get("X_test")

        if state.get("error") or not isinstance(X_train_orig, pd.DataFrame) or not isinstance(y_train_orig, pd.Series):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid X_train/y_train.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train or y_train is missing/invalid for Mutual Information."
            return state
        
        X_train, y_train = X_train_orig.copy(), y_train_orig.copy()
        X_val = X_val_orig.copy() if isinstance(X_val_orig, pd.DataFrame) else None
        X_test = X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else None
        
        try:
            # Ensure no NaNs before mutual_info_classif
            if X_train.isnull().sum().sum() > 0:
                self.logger.warning(f"{tool_name}: X_train contains NaNs before Mutual Information. Attempting median fill for numeric columns.")
                for col in X_train.select_dtypes(include=np.number).columns:
                    if X_train[col].isnull().any(): X_train[col].fillna(X_train[col].median(), inplace=True)
                if X_train.isnull().sum().sum() > 0: raise ValueError("NaNs remain in X_train for Mutual Information.")

            if X_train.empty:
                self.logger.warning(f"{tool_name}: X_train is empty. Skipping mutual information.")
                state["features"] = []
                state = self._log_action(tool_name, "Mutual Information Selection", "Skipped - X_train is empty", state)
                return state

            info_gains = mutual_info_classif(X_train, y_train, random_state=self.seed)
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': info_gains
            }).sort_values('importance', ascending=False)
            
            # Ensure num_top_features is not more than available features
            actual_num_top_features = min(num_top_features, len(X_train.columns))
            top_features = feature_importance.head(actual_num_top_features)['feature'].tolist()

            state["X_train"] = X_train[top_features]
            if X_val is not None: state["X_val"] = X_val[top_features]
            if X_test is not None: state["X_test"] = X_test[top_features]
            state["features"] = top_features
            
            outcome = f"Selected top {len(top_features)} features based on mutual information: {top_features}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, "Mutual Information Selection", outcome, state)
        except Exception as e:
            error_msg = f"Error during Mutual Information selection: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state["X_train"], state["X_val"], state["X_test"] = X_train_orig, X_val_orig, X_test_orig
            state["features"] = list(X_train_orig.columns) if X_train_orig is not None else []
            state = self._log_action(tool_name, "Mutual Information Selection", f"Error: {str(e)}", state)
        return state
    # --- End of Feature Selection Tool Implementations ---

    # --- Model Training Tool Implementation ---
    def _tool_train_model(self, state: MLState, model_type: str = "LogisticRegression", model_params: Optional[Dict[str, Any]] = None) -> MLState:
        tool_name = f"Tool:train_model ({model_type})"
        X_train = state.get("X_train")
        y_train = state.get("y_train")

        if state.get("error") or not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing/invalid X_train/y_train.")
            if not state.get("error"): state["error"] = f"{tool_name}: X_train or y_train is missing/invalid for model training."
            return self._log_action(tool_name, "Train Model", f"Skipped: {state.get('error', 'Missing data')}", state)

        try:
            if model_type == "LogisticRegression":
                params = {
                    "random_state": self.seed,
                    "C": 1,
                    "penalty": 'l2',
                    "solver": 'liblinear',
                    "class_weight": 'balanced',
                    "max_iter": 1000000,
                    **(model_params or {}) # Allow overriding defaults
                }
                model = LogisticRegression(**params)
            elif model_type == "RandomForestClassifier":
                params = {
                    "random_state": self.seed,
                    "n_estimators": 100, # Example default
                    "class_weight": "balanced",
                     **(model_params or {})
                }
                model = RandomForestClassifier(**params)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            model.fit(X_train, y_train)
            
            state["model"] = model
            outcome = f"Trained {model_type} model on {len(X_train)} samples with {len(X_train.columns)} features. Params: {params}"
            self.logger.info(f"{tool_name}: {outcome}")
            state = self._log_action(tool_name, f"Train {model_type}", outcome, state)
            
        except Exception as e:
            error_msg = f"Error training {model_type} model: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, f"Train {model_type}", f"Error: {str(e)}", state)
        return state
    # --- End of Model Training Tool Implementation ---

    # --- Model Evaluation Tool Implementation ---
    def _tool_evaluate_model(self, state: MLState) -> MLState:
        tool_name = "Tool:evaluate_model"
        model = state.get("model")
        X_val, y_val = state.get("X_val"), state.get("y_val")
        X_test, y_test = state.get("X_test"), state.get("y_test")

        if state.get("error") or model is None or \
           not isinstance(X_val, pd.DataFrame) or not isinstance(y_val, pd.Series) or \
           not isinstance(X_test, pd.DataFrame) or not isinstance(y_test, pd.Series):
            self.logger.warning(f"{tool_name}: Skipping due to previous error or missing model/data.")
            if not state.get("error"): state["error"] = f"{tool_name}: Missing model or validation/test data for evaluation."
            return self._log_action(tool_name, "Evaluate Model", f"Skipped: {state.get('error', 'Missing model/data')}", state)

        try:
            metrics = {"validation": {}, "test": {}}
            
            # Validation
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]
            metrics["validation"]["accuracy"] = accuracy_score(y_val, val_pred)
            metrics["validation"]["auc"] = roc_auc_score(y_val, val_proba)
            metrics["validation"]["report"] = classification_report(y_val, val_pred, output_dict=True)
            
            val_outcome = f"Validation - Accuracy: {metrics['validation']['accuracy']:.4f}, AUC: {metrics['validation']['auc']:.4f}"
            self.logger.info(f"{tool_name}: {val_outcome}")
            
            # Test
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1]
            metrics["test"]["accuracy"] = accuracy_score(y_test, test_pred)
            metrics["test"]["auc"] = roc_auc_score(y_test, test_proba)
            metrics["test"]["report"] = classification_report(y_test, test_pred, output_dict=True)

            test_outcome = f"Test - Accuracy: {metrics['test']['accuracy']:.4f}, AUC: {metrics['test']['auc']:.4f}"
            self.logger.info(f"{tool_name}: {test_outcome}")
            
            state["metrics"] = metrics
            overall_outcome = f"Evaluation complete. Val AUC: {metrics['validation']['auc']:.4f}, Test AUC: {metrics['test']['auc']:.4f}"
            state = self._log_action(tool_name, "Evaluate Model", overall_outcome, state)
            
        except Exception as e:
            error_msg = f"Error evaluating model: {str(e)}"
            self.logger.error(f"{tool_name}: {error_msg}", exc_info=True)
            state["error"] = error_msg
            state = self._log_action(tool_name, "Evaluate Model", f"Error: {str(e)}", state)
        return state
    # --- End of Model Evaluation Tool Implementation ---
    # --- End of Feature Selection Tool Implementations ---

    def _log_action(self, agent: str, task: str, outcome: str, state: MLState) -> MLState:
        """Log agent actions"""
        log_entry = f"{datetime.now().isoformat()} | {agent} | {task} | {outcome}"
        
        # Write to log file
        log_file = Path("logs") / "agent_actions.txt"
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"{log_entry}\n")
        
        self.logger.info(f"Agent: {agent} | Task: {task} | Outcome: {outcome}")
        
        # Update state
        if "logs" not in state:
            state["logs"] = []
        state["logs"].append(log_entry)
        return state

    def data_loading_agent(self, state: MLState) -> MLState:
        """Agent responsible for loading and initial data exploration using tools."""
        agent_name = "DataLoadingAgent"
        self.logger.info(f"{agent_name}: Starting data loading and initial exploration.")
        
        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state

        try:
            # Step 1: Load data using the tool
            state = self.tool_executor.execute_tool("load_data", state, data_path=self.data_path)
            if state.get("error"): # Check for error after tool execution
                self.logger.error(f"{agent_name}: Failed during 'load_data' tool execution. Error: {state['error']}")
                # _log_action is called by execute_tool on error, and by the tool itself.
                # Agent might log its own failure to orchestrate.
                # For now, rely on tool executor's logging for tool failure.
                return state

            # Step 2: Perform initial exploration using the tool
            state = self.tool_executor.execute_tool("initial_exploration", state)
            if state.get("error"): # Check for error after tool execution
                self.logger.error(f"{agent_name}: Failed during 'initial_exploration' tool execution. Error: {state['error']}")
                return state
            
            state["current_step"] = "data_loaded_and_explored" # Updated step name
            self.logger.info(f"{agent_name}: Successfully completed data loading and initial exploration.")
            # Log overall agent task success
            state = self._log_action(agent_name, "Data Loading & Exploration Task", "Completed successfully", state)

        except Exception as e: # Catch errors in agent's orchestration logic itself
            error_msg = f"{agent_name}: Orchestration error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Data Loading & Exploration Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def eda_agent(self, state: MLState) -> MLState:
        """Agent responsible for Exploratory Data Analysis using LLM-driven tool orchestration."""
        agent_name = "EDAAgent_LLM"
        self.logger.info(f"{agent_name}: Starting LLM-driven Exploratory Data Analysis.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("data") is None:
            self.logger.error(f"{agent_name}: No data found in state to perform EDA.")
            state["error"] = f"{agent_name}: No data available for EDA."
            return self._log_action(agent_name, "EDA Task", "No data available", state)
        
        if self.llm is None:
            self.logger.error(f"{agent_name}: LLM not initialized. Cannot perform LLM-driven EDA.")
            state["error"] = f"{agent_name}: LLM not available."
            return self._log_action(agent_name, "EDA Task", "LLM not available", state)

        # Tools available to this agent
        eda_tool_names = ["analyze_numeric_features", "detect_fare_outliers", "analyze_target_distribution"]
        available_tools_desc_list = []
        for tool_name in eda_tool_names:
            if tool_name in self.tool_executor.tools:
                tool_obj = self.tool_executor.tools[tool_name]
                available_tools_desc_list.append(f"- {tool_obj.name}: {tool_obj.description}")
            else:
                self.logger.warning(f"{agent_name}: EDA tool '{tool_name}' not found in ToolExecutor.")
        
        if not available_tools_desc_list:
            state["error"] = f"{agent_name}: No EDA tools available for LLM selection."
            self.logger.error(state["error"])
            return self._log_action(agent_name, "EDA Task", "No tools available for LLM", state)
        
        available_tools_str = "\n".join(available_tools_desc_list)

        if state.get("agent_scratchpad") is None: # Ensure scratchpad exists
            state["agent_scratchpad"] = []

        max_eda_steps = 3  # Limit LLM iterations
        executed_eda_tools_count = 0

        for i in range(max_eda_steps):
            self.logger.info(f"{agent_name}: LLM EDA Iteration {i+1}/{max_eda_steps}")

            data_summary = f"Data has shape {state['data'].shape}. Columns: {list(state['data'].columns)}."
            
            previous_actions_summary = "Previously executed EDA tools in this session: "
            # Filter tool_calls for successful EDA tools executed by this agent or similar context
            successful_eda_tool_calls = [
                tc for tc in state.get("tool_calls", [])
                if tc["tool_name"] in eda_tool_names and tc["status"] == "success"
            ]
            if successful_eda_tool_calls:
                previous_actions_summary += ", ".join(
                    [f"{tc['tool_name']} (Result: {tc.get('result_summary', 'completed')})" for tc in successful_eda_tool_calls]
                )
            else:
                previous_actions_summary += "None yet."

            prompt_messages = [
                SystemMessage(content=(
                    "You are an expert data scientist performing Exploratory Data Analysis (EDA) on the Titanic dataset. "
                    "Your goal is to understand the data better before feature engineering and modeling. "
                    "Based on the current data summary, previously executed EDA tools, and available tools, "
                    "decide which single EDA tool to run next. "
                    "The available tools are:\n" + available_tools_str + "\n"
                    "Respond with only the exact name of the tool to run (e.g., 'analyze_numeric_features'). "
                    "If you believe sufficient EDA has been performed for now (e.g., all relevant tools used or 2-3 diverse analyses done), "
                    "or no more listed tools are relevant at this stage, respond with 'DONE'."
                )),
                HumanMessage(content=(
                    f"Current data summary: {data_summary}\n"
                    f"{previous_actions_summary}\n"
                    "Which EDA tool should be run next, or are we DONE?"
                ))
            ]
            
            state["agent_scratchpad"].extend(prompt_messages)

            try:
                llm_response: AIMessage = self.llm.invoke(prompt_messages)
                state["agent_scratchpad"].append(llm_response)
                chosen_tool_name = llm_response.content.strip()
                self.logger.info(f"{agent_name}: LLM chose: '{chosen_tool_name}'")

                if chosen_tool_name == "DONE":
                    self.logger.info(f"{agent_name}: LLM indicated EDA is complete or no more relevant tools.")
                    break
                
                if chosen_tool_name in eda_tool_names:
                    # Check if tool already successfully run to avoid repetition unless LLM has a good reason (not handled here)
                    if any(tc["tool_name"] == chosen_tool_name and tc["status"] == "success" for tc in successful_eda_tool_calls):
                        self.logger.info(f"{agent_name}: LLM chose tool '{chosen_tool_name}' which was already successfully run. Considering it a no-op for this iteration.")
                        state = self._log_action(agent_name, f"LLM EDA: Choose {chosen_tool_name}", "Skipped - already run successfully", state)
                        # If we want to stop if LLM repeats, we could break here or count no-ops.
                        # For now, we let it try another tool in the next iteration if max_steps not reached.
                        continue # Try to get a different tool in the next iteration

                    state = self.tool_executor.execute_tool(chosen_tool_name, state) # These tools don't take extra args
                    if state.get("error"):
                        self.logger.error(f"{agent_name}: Error executing LLM-chosen tool '{chosen_tool_name}': {state['error']}. Halting EDA.")
                        # Error already logged by executor. Agent logs its decision to halt.
                        return self._log_action(agent_name, f"LLM EDA: Execute {chosen_tool_name}", f"Failed: {state['error']}", state)
                    else:
                        executed_eda_tools_count +=1
                        state = self._log_action(agent_name, f"LLM EDA: Execute {chosen_tool_name}", "Success", state)
                else:
                    self.logger.warning(f"{agent_name}: LLM chose an unknown or invalid tool: '{chosen_tool_name}'. Skipping this step.")
                    state = self._log_action(agent_name, "LLM EDA: Tool Selection", f"LLM chose invalid tool: {chosen_tool_name}", state)

            except Exception as e:
                error_msg = f"{agent_name}: Error during LLM interaction or tool execution planning: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                state["error"] = error_msg
                return self._log_action(agent_name, "LLM EDA Task", f"Orchestration Error: {str(e)}", state)
        
        if executed_eda_tools_count == 0 and not state.get("error"):
             self.logger.info(f"{agent_name}: No new EDA tools were executed by the LLM.")
             # This could be because all tools were already run, or LLM decided DONE immediately.

        state["current_step"] = "eda_complete_llm"
        self.logger.info(f"{agent_name}: LLM-driven Exploratory Data Analysis finished after {i+1} iterations. Executed {executed_eda_tools_count} tools.")
        return self._log_action(agent_name, "LLM EDA Task", f"Completed. Executed {executed_eda_tools_count} tools.", state)

    def feature_engineering_agent(self, state: MLState) -> MLState:
        """Agent responsible for feature engineering using tools."""
        agent_name = "FeatureEngineeringAgent"
        self.logger.info(f"{agent_name}: Starting feature engineering.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("data") is None:
            self.logger.error(f"{agent_name}: No data found in state for feature engineering.")
            state["error"] = f"{agent_name}: No data available for feature engineering."
            state = self._log_action(agent_name, "Feature Engineering Task", "No data available", state)
            return state

        try:
            # Tool 1: Drop leakage columns
            columns_to_drop_leakage = ['boat', 'body']
            state = self.tool_executor.execute_tool("drop_features", state, columns=columns_to_drop_leakage, agent_name_context=agent_name)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'drop_features' (leakage). Halting."); return state

            # Tool 2: Impute home.dest with ticket
            state = self.tool_executor.execute_tool("impute_home_dest_with_ticket", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'impute_home_dest_with_ticket'. Halting."); return state
            
            # Tool 3: Extract title from name
            state = self.tool_executor.execute_tool("extract_title_from_name", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'extract_title_from_name'. Halting."); return state

            # Tool 4: Create ticket count feature
            state = self.tool_executor.execute_tool("create_ticket_count_feature", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'create_ticket_count_feature'. Halting."); return state

            # Tool 5: Drop original columns used for feature creation
            columns_to_drop_originals = ['name', 'ticket'] # 'cabin' is handled by its own tool
            state = self.tool_executor.execute_tool("drop_features", state, columns=columns_to_drop_originals, agent_name_context=agent_name)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'drop_features' (originals). Halting."); return state
            
            state["current_step"] = "feature_engineering_complete"
            self.logger.info(f"{agent_name}: Successfully completed feature engineering.")
            state = self._log_action(agent_name, "Feature Engineering Task", "Completed successfully", state)
            
        except Exception as e: # Catch errors in agent's orchestration logic
            error_msg = f"{agent_name}: Orchestration error during Feature Engineering: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Feature Engineering Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def data_imputation_agent(self, state: MLState) -> MLState:
        """Agent responsible for handling missing values using tools."""
        agent_name = "DataImputationAgent"
        self.logger.info(f"{agent_name}: Starting data imputation.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("data") is None and not (state.get("X_train") and state.get("X_val") and state.get("X_test")):
             # Check if either raw data or split data is missing
            self.logger.error(f"{agent_name}: No data found in state for imputation.")
            state["error"] = f"{agent_name}: No data available for imputation."
            state = self._log_action(agent_name, "Data Imputation Task", "No data available", state)
            return state

        try:
            # Tool 1: Split data (if not already split)
            # This tool now populates X_train, X_val, X_test, y_train, y_val, y_test in state
            if not (state.get("X_train") is not None and state.get("y_train") is not None): # Check if split already happened
                state = self.tool_executor.execute_tool("split_data", state)
                if state.get("error"): self.logger.error(f"{agent_name}: Error during 'split_data'. Halting."); return state
            else:
                self.logger.info(f"{agent_name}: Data already split, proceeding with imputation on existing splits.")

            # Tool 2: Impute age
            state = self.tool_executor.execute_tool("impute_age_by_group", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'impute_age_by_group'. Halting."); return state

            # Tool 3: Impute fare
            state = self.tool_executor.execute_tool("impute_fare_median", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'impute_fare_median'. Halting."); return state

            # Tool 4: Impute embarked
            state = self.tool_executor.execute_tool("impute_embarked_mode", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'impute_embarked_mode'. Halting."); return state

            # Tool 5: Extract deck from cabin
            state = self.tool_executor.execute_tool("extract_deck_from_cabin", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'extract_deck_from_cabin'. Halting."); return state
            
            # Tool 6: Impute deck using KNN
            state = self.tool_executor.execute_tool("impute_deck_knn", state)
            if state.get("error"): self.logger.error(f"{agent_name}: Error during 'impute_deck_knn'. Halting."); return state

            # Ensure all X_train, X_val, X_test are pandas DataFrames after operations
            for df_name in ["X_train", "X_val", "X_test"]:
                if not isinstance(state.get(df_name), pd.DataFrame):
                    self.logger.error(f"{agent_name}: {df_name} is not a DataFrame after imputation. Type: {type(state.get(df_name))}")
                    state["error"] = f"{agent_name}: {df_name} became invalid type after imputation."
                    return self._log_action(agent_name, "Data Imputation Task", f"Error: {df_name} invalid type", state)


            state["current_step"] = "imputation_complete"
            self.logger.info(f"{agent_name}: Successfully completed data imputation.")
            state = self._log_action(agent_name, "Data Imputation Task", "Completed successfully", state)
            
        except Exception as e: # Catch errors in agent's orchestration logic itself
            error_msg = f"{agent_name}: Orchestration error during Data Imputation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Data Imputation Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def preprocessing_agent(self, state: MLState) -> MLState:
        """Agent responsible for encoding and scaling using tools."""
        agent_name = "PreprocessingAgent"
        self.logger.info(f"{agent_name}: Starting preprocessing (encoding and scaling).")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if not all(state.get(key) is not None and isinstance(state.get(key), pd.DataFrame) for key in ["X_train", "X_val", "X_test"]):
            self.logger.error(f"{agent_name}: Missing or invalid X_train, X_val, or X_test for preprocessing.")
            state["error"] = f"{agent_name}: Missing or invalid data splits for preprocessing."
            state = self._log_action(agent_name, "Preprocessing Task", "Missing or invalid data splits", state)
            return state

        try:
            # Tool 1: Group rare home.dest values
            state = self.tool_executor.execute_tool("group_rare_home_dest", state)
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'group_rare_home_dest'. Halting preprocessing. Error: {state['error']}")
                return state

            # Tool 2: One-hot encoding
            categorical_cols_to_encode = ['embarked', 'deck', 'home.dest', 'title', 'sex']
            state = self.tool_executor.execute_tool("one_hot_encode_features", state, categorical_cols=categorical_cols_to_encode)
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'one_hot_encode_features'. Halting preprocessing. Error: {state['error']}")
                return state
            
            # Tool 3: Scale continuous features
            continuous_cols_to_scale = ['age', 'fare']
            state = self.tool_executor.execute_tool("scale_features", state, cols_to_scale=continuous_cols_to_scale, scaler_type="StandardScaler")
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'scale_features' (StandardScaler). Halting preprocessing. Error: {state['error']}")
                return state

            # Tool 4: Scale discrete features
            discrete_cols_to_scale = ['pclass', 'sibsp', 'parch', 'ticket_count']
            state = self.tool_executor.execute_tool("scale_features", state, cols_to_scale=discrete_cols_to_scale, scaler_type="MinMaxScaler")
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'scale_features' (MinMaxScaler). Halting preprocessing. Error: {state['error']}")
                return state

            state["current_step"] = "preprocessing_complete"
            self.logger.info(f"{agent_name}: Successfully completed preprocessing.")
            state = self._log_action(agent_name, "Preprocessing Task", "Completed successfully", state)
            
        except Exception as e: # Catch errors in agent's orchestration logic itself
            error_msg = f"{agent_name}: Orchestration error during Preprocessing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Preprocessing Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def feature_selection_agent(self, state: MLState) -> MLState:
        """Agent responsible for feature selection using tools."""
        agent_name = "FeatureSelectionAgent"
        self.logger.info(f"{agent_name}: Starting feature selection.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("X_train") is None or state.get("y_train") is None:
            self.logger.error(f"{agent_name}: Missing X_train or y_train for feature selection.")
            state["error"] = f"{agent_name}: Missing X_train or y_train."
            state = self._log_action(agent_name, "Feature Selection Task", "Missing X_train or y_train", state)
            return state

        try:
            # Tool 1: Balance data using ADASYN
            # This tool modifies X_train and y_train in place in the state
            state = self.tool_executor.execute_tool("balance_data_adasyn", state)
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'balance_data_adasyn'. Halting feature selection. Error: {state['error']}")
                return state

            # Tool 2: Variance Threshold selection
            # This tool modifies X_train, X_val, X_test, and features in state
            variance_threshold_value = 0.01 # Could be decided by an LLM
            state = self.tool_executor.execute_tool("variance_threshold_select", state, threshold=variance_threshold_value)
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'variance_threshold_select'. Halting feature selection. Error: {state['error']}")
                return state
            
            # Tool 3: Mutual Information selection
            # This tool also modifies X_train, X_val, X_test, and features in state
            num_mutual_info_features = 15 # Could be decided by an LLM
            state = self.tool_executor.execute_tool("mutual_information_select", state, num_top_features=num_mutual_info_features)
            if state.get("error"):
                self.logger.error(f"{agent_name}: Error during 'mutual_information_select'. Halting feature selection. Error: {state['error']}")
                return state

            state["current_step"] = "feature_selection_complete"
            self.logger.info(f"{agent_name}: Successfully completed feature selection. Selected {len(state.get('features', []))} features.")
            state = self._log_action(agent_name, "Feature Selection Task", f"Completed successfully. Features: {state.get('features', [])}", state)
            
        except Exception as e: # Catch errors in agent's orchestration logic
            error_msg = f"{agent_name}: Orchestration error during Feature Selection: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Feature Selection Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def training_agent(self, state: MLState) -> MLState:
        """Agent responsible for model training using tools."""
        agent_name = "TrainingAgent"
        self.logger.info(f"{agent_name}: Starting model training.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("X_train") is None or state.get("y_train") is None:
            self.logger.error(f"{agent_name}: Missing X_train or y_train for model training.")
            state["error"] = f"{agent_name}: Missing X_train or y_train."
            state = self._log_action(agent_name, "Model Training Task", "Missing X_train or y_train", state)
            return state

        try:
            # Define model parameters - an LLM could potentially choose/tune these
            model_type = "LogisticRegression" # Could be decided by an LLM
            model_params = {
                "C": 1.0, # Example: LLM could suggest this
                "penalty": 'l2',
                "solver": 'liblinear',
                "class_weight": 'balanced',
                "max_iter": 100000 # Reduced for faster example, LLM could adjust
            }
            
            # Use the train_model tool
            state = self.tool_executor.execute_tool(
                "train_model",
                state,
                model_type=model_type,
                model_params=model_params
            )

            if state.get("error"):
                self.logger.error(f"{agent_name}: Failed during 'train_model' tool execution. Error: {state['error']}")
                # Error already logged by tool executor or tool itself
                return state
            
            state["current_step"] = "training_complete"
            self.logger.info(f"{agent_name}: Successfully completed model training.")
            state = self._log_action(agent_name, "Model Training Task", "Completed successfully", state)

        except Exception as e: # Catch errors in agent's orchestration logic
            error_msg = f"{agent_name}: Orchestration error during model training: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Model Training Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def evaluation_agent(self, state: MLState) -> MLState:
        """Agent responsible for model evaluation using tools."""
        agent_name = "EvaluationAgent"
        self.logger.info(f"{agent_name}: Starting model evaluation.")

        if state.get("error"):
            self.logger.warning(f"{agent_name}: Skipping due to pre-existing error: {state['error']}")
            return state
        if state.get("model") is None or \
           state.get("X_val") is None or state.get("y_val") is None or \
           state.get("X_test") is None or state.get("y_test") is None:
            self.logger.error(f"{agent_name}: Missing model or data splits for evaluation.")
            state["error"] = f"{agent_name}: Missing model or data splits."
            state = self._log_action(agent_name, "Model Evaluation Task", "Missing model or data splits", state)
            return state

        try:
            # Use the evaluate_model tool
            state = self.tool_executor.execute_tool("evaluate_model", state)

            if state.get("error"):
                self.logger.error(f"{agent_name}: Failed during 'evaluate_model' tool execution. Error: {state['error']}")
                # Error already logged by tool executor or tool itself
                return state
            
            state["current_step"] = "evaluation_complete"
            self.logger.info(f"{agent_name}: Successfully completed model evaluation.")
            # The tool itself logs detailed metrics, agent logs overall success
            state = self._log_action(agent_name, "Model Evaluation Task", f"Completed successfully. Test AUC: {state['metrics'].get('test', {}).get('auc', 0):.4f}", state)

        except Exception as e: # Catch errors in agent's orchestration logic
            error_msg = f"{agent_name}: Orchestration error during model evaluation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            state["error"] = error_msg
            state = self._log_action(agent_name, "Model Evaluation Task", f"Orchestration Error: {str(e)}", state)
            
        return state

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(MLState)
        
        # Add nodes
        workflow.add_node("data_loading", self.data_loading_agent)
        workflow.add_node("eda", self.eda_agent)
        workflow.add_node("feature_engineering", self.feature_engineering_agent)
        workflow.add_node("data_imputation", self.data_imputation_agent)
        workflow.add_node("preprocessing", self.preprocessing_agent)
        workflow.add_node("feature_selection", self.feature_selection_agent)
        workflow.add_node("training", self.training_agent)
        workflow.add_node("evaluation", self.evaluation_agent)
        
        # Define edges
        workflow.add_edge("data_loading", "eda")
        workflow.add_edge("eda", "feature_engineering")
        workflow.add_edge("feature_engineering", "data_imputation")
        workflow.add_edge("data_imputation", "preprocessing")
        workflow.add_edge("preprocessing", "feature_selection")
        workflow.add_edge("feature_selection", "training")
        workflow.add_edge("training", "evaluation")
        workflow.add_edge("evaluation", END)
        
        # Set entry point
        workflow.set_entry_point("data_loading")
        
        return workflow.compile()

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        self.logger.info("Starting Agentic ML Pipeline")
        
        # Initialize state
        initial_state = MLState(
            data=None,
            X_train=None,
            X_val=None,
            X_test=None,
            y_train=None,
            y_val=None,
            y_test=None,
            model=None,
            metrics={},
            features=[],
            logs=[],
            current_step="initialized",
            error=None,
            config=self.config.copy(), # New
            iteration_count=0,         # New
            tool_calls=[],             # New
            agent_scratchpad=[]        # New: Initialize agent_scratchpad
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        if final_state.get("error"):
            self.logger.error(f"Pipeline failed: {final_state['error']}")
        else:
            self.logger.info("Pipeline completed successfully")
            self.logger.info(f"Final metrics: {final_state['metrics']}")
        
        return final_state


def main():
    """Main function to run the agentic ML pipeline"""
    # Check if data file exists
    if not os.path.exists("titanic3.xls"):
        print("Error: titanic3.xls not found. Please ensure the dataset is in the current directory.")
        return
    
    # Create and run pipeline
    pipeline = AgenticMLPipeline()
    result = pipeline.run_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("AGENTIC ML PIPELINE SUMMARY")
    print("="*50)
    
    if result.get("error"):
        print(f"❌ Pipeline failed: {result['error']}")
    else:
        print("✅ Pipeline completed successfully!")
        
        if "metrics" in result and result["metrics"]:
            print(f"\n📊 Model Performance:")
            val_metrics = result["metrics"].get("validation", {})
            test_metrics = result["metrics"].get("test", {})
            
            if val_metrics:
                print(f"   Validation - Accuracy: {val_metrics.get('accuracy', 0):.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
            if test_metrics:
                print(f"   Test - Accuracy: {test_metrics.get('accuracy', 0):.4f}, AUC: {test_metrics.get('auc', 0):.4f}")
        
        if "features" in result and result["features"]:
            print(f"\n🔧 Selected Features ({len(result['features'])}):")
            for feature in result["features"]:
                print(f"   - {feature}")
    
    print(f"\n📝 Detailed logs saved to: logs/agent_actions.txt")
    print("="*50)


if __name__ == "__main__":
    main()
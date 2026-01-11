# Interactive Menu Guide

## Overview
The interactive menu has been **restored and enhanced** with all the new Core Code validation and quantity plotting functionality.

## Features Available

### 🎛️ **Main Interactive Menu (17 Options)**

When you run `python main.py`, you'll see an interactive menu with these options:

1. **Enter Start Date** - Configure analysis start date
2. **Enter End Date** - Configure analysis end date  
3. **Enter Product Codes or Ranges** - Filter by specific codes (e.g., "WL100-WL119")
4. **Edit Contamination Parameter** - Outlier detection sensitivity
5. **Edit Max Trendlines** - Number of trend lines on charts
6. **Toggle Data Labels** - Show/hide percentage labels on charts
7. **Edit Max Lines for Labels** - Limit label display
8. **Edit Max Lines for Legend** - Limit legend entries
9. **Edit 'Other' Threshold** - Pie chart grouping threshold
10. **Toggle Info Box Options** - Statistical summary boxes
11. **Toggle Advanced Attribute Filtering** - Filter by Size, Color, etc.
12. **Toggle Save Figures** - Save charts to files
13. **Toggle Skip Transformation** - Use pre-processed data
14. **Toggle Download Data** - Export CSV data
15. **Save Current Parameters** - Store settings for reuse
16. **Load Saved Parameters** - Restore previous settings
17. **DONE - Proceed to Analysis** - Start processing

### 🏷️ **Category Selection Menu**

After main parameters, you'll choose which product categories to analyze:
- Wall Light
- Pendant  
- Chandelier
- Hanging Lantern
- Ceiling Light
- Table Lamp
- Floor Lamp
- Custom
- **Options**: Select specific categories, 'all', or 'none'

### 🔧 **Advanced Attribute Filtering**

If enabled, filter by additional attributes:
- **Size**: Extra Small, Small, Medium, Large, Extra Large, Giant
- **Finish**: Chrome, Bronze, Gold, etc.
- **Colour**: Silver, Bronze, Gold, etc. 
- **Region**: UK, EU, US, etc.
- **Glass**: Clear, Frosted, Amber, etc.
- **Type**: Standard, Premium, etc.
- **IP44**: Yes/No (water resistance)

## New Features Integrated

### ✅ **Automatic Core Code Validation**
During processing, you'll see:
```
============================================================
CORE CODE VALIDATION REPORT
============================================================

Core Code:
  Sales Dataset: 175 unique codes
  Quantity Dataset: 175 unique codes
  ✅ Perfect match - all codes present in both datasets
```

### ✅ **Display Rules Enforcement**
```
📊 CORE CODE DISPLAY RULES:
   • Core Code: Showing full codes for 175 unique entries
   • Core Code 4: Truncated to 4 digits, 45 unique prefixes
   • Top 5 prefixes by count:
     - WL10: 12 items
     - PL34: 8 items
```

### ✅ **Enhanced Chart Options**
New chart types available:
- **Quantity Distribution by Core Code (Pie Chart)**
- **Quantity Distribution by Core Code 4 (Pie Chart)**  
- **Yearly Quantity Trend by Core Code**
- **Quarterly Quantity Trend by Core Code**
- **Monthly Quantity Trend by Core Code**
- **Yearly Quantity Trend by Core Code 4**
- **Quarterly Quantity Trend by Core Code 4**
- **Monthly Quantity Trend by Core Code 4**

## Usage Workflow

### 1. **Start the Program**
```bash
cd "HFL-Analysis-Refactored"
python main.py
```

### 2. **Configure Parameters**
- Use the interactive menu to set your preferences
- Configure date ranges, filtering, and display options
- Save settings for future use

### 3. **Select Categories**
- Choose which product categories to analyze
- Select 'all' for comprehensive analysis
- Or pick specific categories for focused analysis

### 4. **Apply Advanced Filters** (Optional)
- Filter by size, finish, color, region, etc.
- Create targeted analysis subsets

### 5. **Review Validation**
- Check the Core Code validation report
- Verify data consistency between Sales and Quantity
- Review display rule enforcement

### 6. **Generate Visualizations**
- All configured charts will be generated automatically
- Both value-based and quantity-based visualizations
- Charts saved to timestamped directories

### 7. **Export Data** (If Enabled)
- CSV files with chart data
- Processed datasets for further analysis

## Benefits

✅ **User-Friendly**: No need to edit code - configure everything through menus
✅ **Flexible**: Easily switch between different analysis scenarios  
✅ **Comprehensive**: Access to all new Core Code and quantity features
✅ **Validated**: Automatic data quality checks with clear reporting
✅ **Reproducible**: Save/load parameter configurations
✅ **Professional**: Clean chart outputs with family name integration

## Example Session

1. Run `python main.py`
2. Set date range: 2023-01-01 to 2023-12-31
3. Enable advanced filtering: Yes
4. Save figures: Yes  
5. Download data: Yes
6. Select categories: Wall Light, Pendant
7. Filter by finish: Chrome, Bronze
8. **DONE** → Analysis runs with validation and generates charts

The program will automatically:
- Validate Core Code consistency
- Enforce display rules
- Generate both value and quantity visualizations
- Include family names in charts
- Save all outputs to organized directories
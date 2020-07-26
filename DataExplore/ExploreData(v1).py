# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:08:56 2020

@author: Horace Yang
"""

import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
from numpy import corrcoef
import os
import sys

# In[1]
fdir = 'dataset/'
fname = 'exp1.xlsx'

sfile = pd.ExcelFile(fdir+fname)
sfile.sheet_names
df = sfile.parse(sfile.sheet_names[2])  # sheet name
# plt.plot(df.iloc[:,0])
# plt.show()f = sfile.parse('第1顆2500_40_51')  # sheet name

# In[2]
# https://pythonspot.com/pyqt5-matplotlib/
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use("Qt5Agg")

# find  matplotlib default font
# https://medium.com/marketingdatascience/解決python-3-matplotlib與seaborn視覺化套件中文顯示問題-f7b3773a889b
from matplotlib.font_manager import FontProperties  

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10.5, height=8.3, dpi=100): 
        #第一步：建立一個建立Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父類中啟用Figure視窗
        super(PlotCanvas,self).__init__(self.fig)
        #第三步：建立一個子圖，用於繪製圖形用，111表示子圖編號，如matlab的subplot(1,1,1)
        self.axes1 = self.fig.add_subplot(211)
        self.axes2 = self.fig.add_subplot(212)

        self.applyfont = FontProperties(fname=r'C:\Anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\微软正黑体.ttf')
        
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data1, title1, data2, title2):
        self.axes1.cla()  #  which clears data and axes
        self.axes1.plot(data1, 'b-')
        self.axes1.set_title(title1, fontproperties=self.applyfont)
        
        self.axes2.cla()  #  which clears data and axes
        self.axes2.plot(data2, 'b-')
        self.axes2.set_title(title2, fontproperties=self.applyfont)
        
        self.draw()
        
# In[3]        
path = os.getcwd()
guiFile = 'ExploreData(v1).ui'
MainWindow, QtBaseClass = uic.loadUiType(guiFile)

class MainUi(QtWidgets.QMainWindow, MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        QtWidgets.QMainWindow.__init__(self)  # initial mainform
        MainWindow.__init__(self)
        self.setupUi(self)
        self.options = QtWidgets.QFileDialog.Options()
        self.options |= QtWidgets.QFileDialog.DontUseNativeDialog
 
        self.btnOverview.clicked.connect(self.overview)   # connect button
        self.btnNextSheet1.clicked.connect(self.nextSheet1)   # connecbutton
        self.btnPrevSheet1.clicked.connect(self.prevSheet1)   # connect button
        self.btnNextSheet2.clicked.connect(self.nextSheet2)   # connecbutton
        self.btnPrevSheet2.clicked.connect(self.prevSheet2)   # connect button
        self.btnNextCol1.clicked.connect(self.nextCol1)   # connect button
        self.btnPrevCol1.clicked.connect(self.prevCol1)   # connect button
        self.btnNextCol2.clicked.connect(self.nextCol2)   # connect button
        self.btnPrevCol2.clicked.connect(self.prevCol2)   # connect button
            
        self.plotchart = PlotCanvas(self,)
        graphicscene = QtWidgets.QGraphicsScene()  #建立一個場景
        graphicscene.addWidget(self.plotchart)  #將圖形元素新增到場景中
        self.graphicsView.setScene(graphicscene)  #將建立新增到圖形檢視顯示視窗
        self.graphicsView.show()

    def overview(self): # when press button
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*)", options=self.options)
        if fileName:
            self.edSourceName.setText(fileName)  
            self.sfile = pd.ExcelFile(fileName)
            self.sheetPt1=0  # default sheet1 no.
            self.sheetPt2=0  # default sheet2 no.
            self.colPt1=0    # default column1 no.
            self.colPt2=0    # default column2 no.
            self.df1 = pd.DataFrame()  # new dataframe1
            self.df2 = pd.DataFrame()  # new dataframe2
        self.RefreshForm()
        
    def nextSheet1(self): 
        if self.sheetPt1 < len(self.sfile.sheet_names): 
            self.sheetPt1 = self.sheetPt1 + 1      
        self.RefreshForm()
    
    def prevSheet1(self): 
        if self.sheetPt1 > 0: 
            self.sheetPt1 = self.sheetPt1 - 1      
        self.RefreshForm()

    def nextSheet2(self): 
        if self.sheetPt2 < len(self.sfile.sheet_names): 
            self.sheetPt2 = self.sheetPt2 + 1      
        self.RefreshForm()
    
    def prevSheet2(self): 
        if self.sheetPt2 > 0: 
            self.sheetPt2 = self.sheetPt2 - 1      
        self.RefreshForm()
        
    def RefreshForm(self):
        sn1 = self.sfile.sheet_names[self.sheetPt1]
        self.edSheetName1.setText(sn1) 
        self.df1 = self.sfile.parse(sn1)  # sheet name
        sample1 = self.df1.iloc[1:,self.colPt1]
        title1 = sn1 + ': ' + str(self.df1.columns[self.colPt1])
        self.edFieldName1.setText(str(self.colPt1)+': '+str(self.df1.columns[self.colPt1]))
        self.edFieldData1.setText(sample1.to_string())
        
        sn2 = self.sfile.sheet_names[self.sheetPt2]
        self.edSheetName2.setText(sn2) 
        self.df2 = self.sfile.parse(sn2)  # sheet name
        sample2 = self.df2.iloc[1:,self.colPt2]
        title2 = sn2 + ': ' + str(self.df2.columns[self.colPt2])
        self.edFieldName2.setText(str(self.colPt2)+': '+str(self.df2.columns[self.colPt2]))
        self.edFieldData2.setText(sample2.to_string())
        
        minlen = min(len(sample1),len(sample2))
        self.edCorr.setText(str(corrcoef(sample1[:minlen-1], sample2[:minlen-1])[1,0])[:5])
        self.plotchart.plot(sample1, title1, sample2, title2)
        
    def nextCol1(self): 
        if self.colPt1 < self.df1.shape[1]: 
            self.colPt1 = self.colPt1 + 1      
        self.RefreshForm()
    
    def prevCol1(self): 
        if self.colPt1 > 0: 
            self.colPt1 = self.colPt1 - 1 
        self.RefreshForm()
    
    def nextCol2(self): 
        if self.colPt2 < self.df2.shape[1]: 
            self.colPt2 = self.colPt2 + 1      
        self.RefreshForm()
    
    def prevCol2(self): 
        if self.colPt2 > 0: 
            self.colPt2 = self.colPt2 - 1 
        self.RefreshForm()
        
# In[]
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = MainUi()
    
    Form.show()    
    sys.exit(app.exec_())

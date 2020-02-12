from django.shortcuts import render
import numpy as np
from .forms import DetailsForm
from . import ccp
# Create your views here.

def DetailView(request):
    submitbutton= request.POST.get("submit")
    form= DetailsForm(request.POST or None)
    output = ''
    inp_arr = ''
    arr = np.arange(25)
    if form.is_valid():
    	
    	age = form.cleaned_data['age']
    	job = form.cleaned_data['job']
    	credit_amount = form.cleaned_data['credit_amount']
    	duration = form.cleaned_data['duration']
    	purpose = form.cleaned_data['purpose']
    	sex = form.cleaned_data['sex']
    	housing = form.cleaned_data['housing']
    	savings_account = form.cleaned_data['savings_account']
    	risk_bad = form.cleaned_data['risk_bad']
    	checking_account = form.cleaned_data['checking_account']
    	age_category = form.cleaned_data['age_category']
    	
    	
    	

    	arr[0] = age
    	if job == 'N':
    		arr[1] = 0
    	elif job == 'B':
    		arr[1] = 1
    	elif job == 'G':
    		arr[1] = 2
    	else:
    		arr[1] = 3

    	arr[2] = credit_amount
    	arr[3] = duration
    	if purpose == 'C':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (1,0,0,0,0,0,0)
    	elif purpose == 'D':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,1,0,0,0,0,0)
    	elif purpose == 'E':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,0,1,0,0,0,0)
    	elif purpose == 'F':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,0,0,1,0,0,0)
    	elif purpose == 'Ra':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,0,0,0,1,0,0)
    	elif purpose == 'Re':
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,0,0,0,0,1,0)
    	else:
    		arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10] = (0,0,0,0,0,0,1)
    	if sex == 'M':
    		arr[11] = 1
    	else:
    		arr[11] = 0
    	if housing == 'O':
    		arr[12],arr[13] = (1,0)
    	else:
    		arr[12],arr[13] = (0,1)
    	if savings_account == 'M':
    		arr[14],arr[15],arr[16],arr[17] = (1,0,0,0)
    	elif savings_account == 'N':
    		arr[14],arr[15],arr[16],arr[17] = (0,1,0,0)
    	elif savings_account == 'Q':
    		arr[14],arr[15],arr[16],arr[17] = (0,0,1,0)
    	else:
    		arr[14],arr[15],arr[16],arr[17] = (0,0,0,1)
    	if risk_bad == 'Y':
    		arr[18] = 1
    	else:
    		arr[18] = 0
    	if checking_account == 'M':
    		arr[19],arr[20],arr[21] = (1,0,0)
    	elif checking_account == 'N':
    		arr[19],arr[20],arr[21] = (0,1,0)
    	else:
    		arr[19],arr[20],arr[21] = (0,0,1)
    	if age_category == 'Y':
    		arr[22],arr[23],arr[24] = (1,0,0)
    	elif age_category == 'A':
    		arr[22],arr[23],arr[24] = (0,1,0)
    	else:
    		arr[22],arr[23],arr[24] = (0,0,1)
    	
    	output = ccp.predict(arr.reshape(1, -1))
    	inp_arr = arr
        
    context= {'form': form, 'output' : output,'submitbutton': submitbutton, 'inp_arr' : inp_arr}
        
    return render(request, 'index.html', context)
        

		


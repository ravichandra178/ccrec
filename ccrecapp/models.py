from django.db import models

# Create your models here.

class Details(models.Model):
	JOB_CHOICES = (
		('N','No'),
		('B','Business'),
		('G','Government Sector'),
		('P','Private Sector'),
	)
	PURPOSE_CHOICES = (
        ('C', 'Car'),
        ('D', 'Domestic'),
        ('E', 'Education'),
        ('F', 'Furniture'),
        ('Ra', 'Radio/TV'),
        ('Re', 'Repairs'),
        ('V', 'Vacation/Others'),
    )
	GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )
	HOUSING_CHOICES = (
        ('O', 'Own'),
        ('R', 'Rent'),
    )
	SAVINGS_CHOICES = (
        ('M', 'Moderate'),
        ('N', 'No-Info'),
        ('Q', 'Quite-Rich'),
        ('R', 'Rich'),
    )
	RISK_CHOICES = (
    	('Y', 'Yes'),
    	('N', 'No'),
    )
	CHECK_CHOICES = (
        ('M', 'Moderate'),
        ('N', 'No-Info'),
        ('R', 'Rich'),
    )
	AGECAT_CHOICES = (
        ('Y', 'Young'),
        ('A', 'Adult'),
        ('S', 'Senior'),
    )
	
	age = models.IntegerField()
	job = models.CharField(max_length = 1,choices = JOB_CHOICES)
	credit_amount = models.IntegerField()
	duration = models.FloatField(max_length = 10)
	purpose = models.CharField(max_length = 20,choices = PURPOSE_CHOICES)
	sex = models.CharField(max_length = 1,choices = GENDER_CHOICES)
	housing = models.CharField(max_length = 1,choices = HOUSING_CHOICES)
	savings_account = models.CharField(max_length = 1,choices = SAVINGS_CHOICES)
	risk_bad = models.CharField(max_length = 1,choices = RISK_CHOICES)
	checking_account = models.CharField(max_length = 1,choices = CHECK_CHOICES)
	age_category = models.CharField(max_length = 1,choices = AGECAT_CHOICES)

	def __str__(self):
		return self.unnamed



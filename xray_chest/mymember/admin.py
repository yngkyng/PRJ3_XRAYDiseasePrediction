from django.contrib import admin
from mymember.models import Member


class MemberAdmin(admin.ModelAdmin):
    list_display = ( "userid", "passwd", "name", "email", "address", "tel")


admin.site.register(Member, MemberAdmin)
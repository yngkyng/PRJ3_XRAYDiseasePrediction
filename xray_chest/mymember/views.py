from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import login as dlogin, logout as dlogout
from mymember.models import Member


def home(request):
    if 'userid' not in request.session.keys():
        return render(request, 'mymember/login.html')
    else:
        return render(request, 'mymember/main.html')

def join(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        name = request.POST['name']
        # email = request.POST['email']
        # address = request.POST['address']
        # tel = request.POST['tel']
        Member(userid=userid, passwd=passwd, name=name,
               # email=email, address=address, tel=tel
               ).save()
        request.session['userid'] = userid
        request.session['name'] = name
        return render(request, 'mymember/main.html')
    else:
        return render(request, 'mymember/join.html')


def login(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        row = Member.objects.filter(userid=userid, passwd=passwd).first()
        if row is not None:
            request.session['userid'] = userid
            request.session['name'] = row.name
            return render(request, 'mymember/main.html')
        else:
            return render(request, 'mymember/login.html', {'msg': '아이디 또는 비밀번호가 일치하지 않습니다. 다시 한번 로그인해 주세요'})
    else:
        return render(request, 'mymember/login.html')


def logout(request):
    request.session.clear()
    return redirect('/mymember')

def custlist(request):
    userid = request.session.get("userid",False)
    print('id:',userid)
    item = Member.objects.get(userid=userid)
    print(item)
    return render(request, "mymember/custlist.html", {"item": item})

def update(request):
    userid = request.POST['userid']
    row_new = Member(userid=userid,
                     passwd=request.POST["passwd"],
                     name=request.POST["name"],
                     email=request.POST["email"],
                     address=request.POST["address"],
                     tel=request.POST["tel"])
    row_new.save()
    request.session['name'] = row_new.name
    return redirect("/mymember/custlist")

def detail(request):
    memb=Member.objects.get(userid=request.GET["userid"])
    return render(request, "mymember/detail.html",{"memb":memb})


def delete(request):
    Member.objects.get(userid=request.POST["userid"]).delete()
    return redirect("/mymember")